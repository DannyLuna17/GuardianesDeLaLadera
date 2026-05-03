"""Generate temporal pseudo-absence labels per the v1 project policy.

The landslide inventory in the DB is purely positive — every
``ZoneOutcomeLabel`` references a real UNGRD event with a
``target_score`` ≥ 0.35. With no negatives, ``beta_regression`` cannot fit
its precision parameter cleanly and ``AUPRC`` collapses to ``None`` in the
benchmark output because there is no class separation.

The agreed policy ``pseudo_absence_temporal_v1`` (memory entry
``project_pseudo_absence_v1.md``, 2026-04-19) prescribes:

- **Sampling**: temporal pseudo-absences per zone.
- **Ratio**: 1 positive : 2 negatives at zone level.
- **Exclusion window**: skip any date within ±14 days of a positive event in
  the same zone.
- **Date domain**: uniformly random within the zone's first-event-date to
  last-event-date window, minus exclusion windows.
- **Target score**: ``0.05`` — strictly inside the ``(0,1)`` open interval
  required by ``beta_regression``.
- **Source tag**: ``pseudo_absence_temporal_v1``. ``severity = "stable"`` so
  the rows are filterable downstream.

Phase 2 (rainfall-conditioned negatives) is deferred until IDEAM
precipitation history lands; do not implement it here.

Idempotent: zones whose ``pseudo_absence_temporal_v1`` source already has
labels are skipped without inserting duplicates.

Usage:

    uv run python scripts/landslide_inventory/generate_pseudo_absences.py

    uv run python scripts/landslide_inventory/generate_pseudo_absences.py \\
        --ratio 2 --exclusion-days 14 --target-score 0.05 \\
        --random-seed 42

"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


_SCRIPT_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _SCRIPT_DIR.parent.parent
_SCRIPTS_DIR = _SCRIPT_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _lib.dates import parse_iso_datetime as _shared_parse_iso  # noqa: E402


PSEUDO_ABSENCE_SOURCE = "pseudo_absence_temporal_v1"
PSEUDO_ABSENCE_SEVERITY = "stable"
PSEUDO_ABSENCE_TARGET_SCORE = 0.05
DEFAULT_RATIO = 2  # negatives per positive
DEFAULT_EXCLUSION_DAYS = 14
DEFAULT_RANDOM_SEED = 42
MAX_SAMPLING_ATTEMPTS_MULTIPLIER = 8
# How many random draws we'll try per requested negative before bailing — at
# 1:2 ratio with 14-day exclusion, a zone whose positives cluster on a few
# days has roughly (window - 28*positives) eligible days; multiplier=8 covers
# the worst realistic case before we declare the zone saturated.


def _parse_iso(value: datetime | str | None) -> datetime | None:
    return _shared_parse_iso(value)


def _excluded(
    candidate: datetime,
    positive_dates: list[datetime],
    exclusion: timedelta,
) -> bool:
    for positive in positive_dates:
        if abs((candidate - positive).total_seconds()) <= exclusion.total_seconds():
            return True
    return False


def run_generator(
    *,
    ratio: int = DEFAULT_RATIO,
    exclusion_days: int = DEFAULT_EXCLUSION_DAYS,
    target_score: float = PSEUDO_ABSENCE_TARGET_SCORE,
    random_seed: int = DEFAULT_RANDOM_SEED,
    dry_run: bool = False,
    cohort_observed_after: datetime | None = None,
    cohort_observed_before: datetime | None = None,
) -> dict:
    from app.core.config import get_settings
    from app.db.session import get_engine, reset_engine_cache
    from app.models.domain import ZoneOutcomeLabel
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    get_settings.cache_clear()
    reset_engine_cache()
    engine = get_engine()

    rng = random.Random(random_seed)
    exclusion = timedelta(days=exclusion_days)

    stats: dict = {
        "policy": PSEUDO_ABSENCE_SOURCE,
        "ratio": ratio,
        "exclusion_days": exclusion_days,
        "target_score": target_score,
        "random_seed": random_seed,
        "zones_with_positives": 0,
        "zones_skipped_already_seeded": 0,
        "zones_skipped_no_room": 0,
        "positives_total": 0,
        "negatives_inserted": 0,
        "negatives_requested": 0,
        "by_zone_sample_examples": [],
    }

    with Session(engine) as session, session.begin():
        positives_by_zone: dict[str, list[ZoneOutcomeLabel]] = defaultdict(list)
        existing_negatives_by_zone: dict[str, int] = defaultdict(int)
        for label in session.scalars(select(ZoneOutcomeLabel)).all():
            if label.source == PSEUDO_ABSENCE_SOURCE:
                existing_negatives_by_zone[label.zone_id] += 1
                continue
            if label.target_score is None or label.target_score <= target_score:
                # Treat low-score rows as not-positive for the purpose of this
                # generator; this protects against an old run leaking back in.
                continue
            positives_by_zone[label.zone_id].append(label)

        stats["positives_total"] = sum(
            len(items) for items in positives_by_zone.values()
        )
        stats["zones_with_positives"] = len(positives_by_zone)
        now = datetime.now(timezone.utc).replace(microsecond=0)

        for zone_id, positives in positives_by_zone.items():
            if existing_negatives_by_zone.get(zone_id, 0) > 0:
                stats["zones_skipped_already_seeded"] += 1
                continue
            positive_dates = sorted(
                _parse_iso(p.observed_at)
                for p in positives
                if _parse_iso(p.observed_at) is not None
            )
            if not positive_dates:
                continue
            window_start = positive_dates[0]
            window_end = positive_dates[-1]
            if cohort_observed_after is not None:
                window_start = max(window_start, cohort_observed_after)
            if cohort_observed_before is not None:
                window_end = min(window_end, cohort_observed_before)
            window_seconds = (window_end - window_start).total_seconds()
            if window_seconds <= 0:
                # All positives on the same day; cannot sample temporally.
                stats["zones_skipped_no_room"] += 1
                continue

            target_count = max(1, ratio * len(positives))
            stats["negatives_requested"] += target_count

            sampled_dates: list[datetime] = []
            attempts = 0
            max_attempts = target_count * MAX_SAMPLING_ATTEMPTS_MULTIPLIER
            while len(sampled_dates) < target_count and attempts < max_attempts:
                attempts += 1
                offset_seconds = rng.uniform(0.0, window_seconds)
                candidate = window_start + timedelta(seconds=offset_seconds)
                # Snap to midnight UTC so sampled dates align with daily UNGRD
                # records and are simpler to read in evidence dumps.
                candidate = candidate.replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                if _excluded(candidate, positive_dates, exclusion):
                    continue
                if any(
                    abs((candidate - prior).total_seconds())
                    <= exclusion.total_seconds()
                    for prior in sampled_dates
                ):
                    continue
                sampled_dates.append(candidate)

            if not sampled_dates:
                stats["zones_skipped_no_room"] += 1
                continue

            if (
                len(stats["by_zone_sample_examples"]) < 5
                and len(sampled_dates) >= 2
            ):
                stats["by_zone_sample_examples"].append(
                    {
                        "zone_id": zone_id,
                        "positives": len(positives),
                        "sampled_negatives": len(sampled_dates),
                        "first_negative": sampled_dates[0].isoformat(),
                        "last_negative": sampled_dates[-1].isoformat(),
                    }
                )

            for index, sampled_at in enumerate(sampled_dates):
                evidence = {
                    "policy": PSEUDO_ABSENCE_SOURCE,
                    "ratio_target": ratio,
                    "exclusion_days": exclusion_days,
                    "window_start": window_start.isoformat(),
                    "window_end": window_end.isoformat(),
                    "rng_seed": random_seed,
                    "sample_index": index,
                    "positive_count_in_zone": len(positives),
                }
                label = ZoneOutcomeLabel(
                    zone_id=zone_id,
                    feature_run_id=None,
                    observed_at=sampled_at,
                    target_score=target_score,
                    source=PSEUDO_ABSENCE_SOURCE,
                    status="confirmed",
                    notes=None,
                    evidence=evidence,
                    created_at=now,
                    updated_at=now,
                )
                if not dry_run:
                    session.add(label)
                stats["negatives_inserted"] += 1

        if dry_run:
            session.rollback()

    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ratio",
        type=int,
        default=DEFAULT_RATIO,
        help=f"Negatives per positive (default: {DEFAULT_RATIO}).",
    )
    parser.add_argument(
        "--exclusion-days",
        type=int,
        default=DEFAULT_EXCLUSION_DAYS,
        help=(
            f"±N day exclusion window around positives "
            f"(default: {DEFAULT_EXCLUSION_DAYS})."
        ),
    )
    parser.add_argument(
        "--target-score",
        type=float,
        default=PSEUDO_ABSENCE_TARGET_SCORE,
        help=(
            f"Score for negatives, must be in the open (0,1) interval "
            f"(default: {PSEUDO_ABSENCE_TARGET_SCORE})."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"RNG seed for reproducibility (default: {DEFAULT_RANDOM_SEED}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute counts and rollback. Do not insert.",
    )
    args = parser.parse_args(argv)

    if not 0 < args.target_score < 1:
        print(
            f"--target-score must be strictly in (0, 1); got {args.target_score}",
            file=sys.stderr,
        )
        return 1

    stats = run_generator(
        ratio=args.ratio,
        exclusion_days=args.exclusion_days,
        target_score=args.target_score,
        random_seed=args.random_seed,
        dry_run=args.dry_run,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
