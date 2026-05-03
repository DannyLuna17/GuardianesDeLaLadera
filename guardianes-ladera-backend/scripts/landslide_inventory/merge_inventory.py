"""Merge per-source normalized JSONL files into the canonical landslide inventory.

Inputs: any number of ``*_normalized.jsonl`` files (UNGRD, SIMMA, DesInventar ...).
All must already follow the shared schema emitted by the per-source normalizers.

The merge applies the dedup rule agreed with the research brief:

    Events within 3 km and 3 days of each other, in the same municipality
    (when municipality is known, else just 3 km + 3 days), are treated as a
    single physical event. Keep the record with the best ``record_quality``,
    merge other event ids into ``evidence.related_event_ids``, and take the
    highest severity among duplicates.

Outputs:
    <out-dir>/colombia_landslide_events_v1.csv
    <out-dir>/colombia_landslide_events_v1.meta.json
    <out-dir>/colombia_landslide_events_v1.report.md

Usage:

    uv run python scripts/landslide_inventory/merge_inventory.py \\
        --input data/inventory/01_staging/ungrd_normalized.jsonl \\
        --input data/inventory/01_staging/simma_normalized.jsonl \\
        --out-dir data/inventory/02_final

"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


_SCRIPT_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _SCRIPT_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _lib.dates import parse_iso_datetime as _parse_iso  # noqa: E402
from _lib.geo import haversine_km as _haversine_km  # noqa: E402


RECORD_QUALITY_RANK = {"high": 3, "medium": 2, "low": 1}
SEVERITY_RANK = {"fatal": 4, "severe": 3, "moderate": 2, "minor": 1}

OUTPUT_CSV_FIELDS = [
    "event_id",
    "source",
    "observed_at",
    "municipality",
    "department",
    "divipola",
    "latitude",
    "longitude",
    "severity",
    "movement_type",
    "deaths",
    "injured",
    "missing",
    "homes_destroyed",
    "homes_damaged",
    "description",
    "source_url",
    "record_quality",
    "related_event_ids",
    "related_sources",
]

DEDUP_DISTANCE_KM = 3.0
DEDUP_TIME_DAYS = 3


def _load_jsonl(path: Path) -> list[dict]:
    events: list[dict] = []
    with open(path, "r", encoding="utf-8") as src:
        for line in src:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def _municipality_key(event: dict) -> str:
    return (event.get("municipality") or "").strip().upper()


def _pick_primary(events: list[dict]) -> dict:
    def sort_key(event: dict) -> tuple:
        return (
            -RECORD_QUALITY_RANK.get(event.get("record_quality") or "", 0),
            -SEVERITY_RANK.get(event.get("severity") or "", 0),
            # SGC preferred over UNGRD when tied because it carries coordinates
            0 if event.get("source") == "SGC_SIMMA" else 1,
            event.get("event_id") or "",
        )

    return sorted(events, key=sort_key)[0]


def _merge_cluster(cluster: list[dict]) -> dict:
    if len(cluster) == 1:
        event = dict(cluster[0])
        event["related_event_ids"] = []
        event["related_sources"] = []
        return event

    primary = dict(_pick_primary(cluster))
    other_events = [event for event in cluster if event is not primary]
    if not other_events:
        # _pick_primary returned a fresh dict; find the real primary object
        primary_id = primary.get("event_id")
        other_events = [
            event for event in cluster if event.get("event_id") != primary_id
        ]
    primary["related_event_ids"] = [
        event.get("event_id") for event in other_events if event.get("event_id")
    ]
    primary["related_sources"] = sorted(
        {event.get("source") for event in other_events if event.get("source")}
    )

    highest_severity = max(
        (event.get("severity") or "minor" for event in cluster),
        key=lambda s: SEVERITY_RANK.get(s, 0),
    )
    primary["severity"] = highest_severity

    # Pull the first non-null scalar we can find across duplicates for each
    # damage field so the merged record does not drop already-known numbers.
    for field in (
        "deaths",
        "injured",
        "missing",
        "homes_destroyed",
        "homes_damaged",
    ):
        if primary.get(field) is None:
            for event in other_events:
                value = event.get(field)
                if value is not None:
                    primary[field] = value
                    break

    if primary.get("latitude") is None or primary.get("longitude") is None:
        for event in other_events:
            lat, lon = event.get("latitude"), event.get("longitude")
            if lat is not None and lon is not None:
                primary["latitude"] = lat
                primary["longitude"] = lon
                break

    return primary


def _bucket_key(event: dict) -> tuple[str, str]:
    observed_at = _parse_iso(event.get("observed_at"))
    day_key = observed_at.date().isoformat() if observed_at else "unknown"
    return _municipality_key(event), day_key


def _are_duplicates(a: dict, b: dict) -> bool:
    ts_a = _parse_iso(a.get("observed_at"))
    ts_b = _parse_iso(b.get("observed_at"))
    if ts_a is None or ts_b is None:
        return False
    delta_days = abs((ts_a - ts_b).total_seconds()) / 86400.0
    if delta_days > DEDUP_TIME_DAYS:
        return False

    muni_a = _municipality_key(a)
    muni_b = _municipality_key(b)
    # If both sides know the municipality and they disagree, not a duplicate.
    if muni_a and muni_b and muni_a != muni_b:
        return False

    lat_a, lon_a = a.get("latitude"), a.get("longitude")
    lat_b, lon_b = b.get("latitude"), b.get("longitude")
    if None not in (lat_a, lon_a, lat_b, lon_b):
        distance = _haversine_km(lat_a, lon_a, lat_b, lon_b)
        return distance <= DEDUP_DISTANCE_KM

    # If one side lacks coordinates, fall back to municipality+day match.
    if muni_a and muni_a == muni_b:
        return True
    return False


def _cluster(events: list[dict]) -> list[list[dict]]:
    # Bucket by (municipality, day); adjacent days are checked within the bucket
    # and with neighbour buckets ±DEDUP_TIME_DAYS to cover the time window.
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for event in events:
        buckets[_bucket_key(event)].append(event)

    neighbour_days: list[int] = list(range(-DEDUP_TIME_DAYS, DEDUP_TIME_DAYS + 1))
    seen: set[int] = set()
    clusters: list[list[dict]] = []

    def _neighbour_candidates(event: dict) -> list[dict]:
        observed_at = _parse_iso(event.get("observed_at"))
        muni = _municipality_key(event)
        if observed_at is None:
            return list(buckets.get((muni, "unknown"), []))
        day = observed_at.date()
        candidates: list[dict] = []
        for offset in neighbour_days:
            key = (
                muni,
                (day.replace() if offset == 0 else _shift_day(day, offset)).isoformat(),
            )
            candidates.extend(buckets.get(key, []))
        return candidates

    id_map = {id(event): event for event in events}
    for event in events:
        if id(event) in seen:
            continue
        frontier = [event]
        cluster: list[dict] = []
        while frontier:
            current = frontier.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            cluster.append(current)
            for candidate in _neighbour_candidates(current):
                if id(candidate) in seen:
                    continue
                if _are_duplicates(current, candidate):
                    frontier.append(candidate)
        # sanity: ensure we only hold real event refs
        cluster = [id_map[id(event)] for event in cluster if id(event) in id_map]
        clusters.append(cluster)
    return clusters


def _shift_day(day, offset_days: int):
    from datetime import timedelta

    return day + timedelta(days=offset_days)


def _write_csv(output_path: Path, events: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=OUTPUT_CSV_FIELDS)
        writer.writeheader()
        for event in events:
            row = {
                key: (
                    ",".join(str(item) for item in (event.get(key) or []))
                    if key in {"related_event_ids", "related_sources"}
                    else event.get(key)
                )
                for key in OUTPUT_CSV_FIELDS
            }
            writer.writerow(row)


def _build_meta(events: list[dict], source_counts: Counter) -> dict:
    by_municipality: Counter = Counter()
    by_department: Counter = Counter()
    by_year: Counter = Counter()
    by_source: Counter = Counter()
    by_quality: Counter = Counter()
    by_severity: Counter = Counter()
    for event in events:
        if event.get("municipality"):
            by_municipality[event["municipality"]] += 1
        if event.get("department"):
            by_department[event["department"]] += 1
        ts = _parse_iso(event.get("observed_at"))
        if ts is not None:
            by_year[str(ts.year)] += 1
        by_source[event.get("source") or "unknown"] += 1
        by_quality[event.get("record_quality") or "unknown"] += 1
        by_severity[event.get("severity") or "unknown"] += 1

    return {
        "total_events": len(events),
        "by_source": dict(by_source),
        "input_rows_by_source": dict(source_counts),
        "by_record_quality": dict(by_quality),
        "by_severity": dict(by_severity),
        "by_year": dict(sorted(by_year.items())),
        "top_municipalities": by_municipality.most_common(25),
        "top_departments": by_department.most_common(20),
        "dedup_distance_km": DEDUP_DISTANCE_KM,
        "dedup_time_days": DEDUP_TIME_DAYS,
    }


def _build_report(meta: dict, inputs: list[Path]) -> str:
    lines: list[str] = []
    lines.append("# Colombia landslide event inventory — v1")
    lines.append("")
    lines.append(f"Total deduplicated events: **{meta['total_events']}**")
    lines.append("")
    lines.append("## Input files")
    for path in inputs:
        lines.append(f"- `{path}`")
    lines.append("")
    lines.append("## By source (post-dedup)")
    for key, value in sorted(meta["by_source"].items(), key=lambda kv: -kv[1]):
        lines.append(f"- `{key}` — {value}")
    lines.append("")
    lines.append("## Input rows (pre-dedup)")
    for key, value in sorted(
        meta["input_rows_by_source"].items(), key=lambda kv: -kv[1]
    ):
        lines.append(f"- `{key}` — {value}")
    lines.append("")
    lines.append("## Record quality")
    for key, value in sorted(
        meta["by_record_quality"].items(), key=lambda kv: -kv[1]
    ):
        lines.append(f"- `{key}` — {value}")
    lines.append("")
    lines.append("## Severity distribution")
    for key, value in sorted(
        meta["by_severity"].items(), key=lambda kv: -kv[1]
    ):
        lines.append(f"- `{key}` — {value}")
    lines.append("")
    lines.append("## Year distribution")
    for key, value in meta["by_year"].items():
        lines.append(f"- `{key}` — {value}")
    lines.append("")
    lines.append("## Top 25 municipalities")
    for name, count in meta["top_municipalities"]:
        lines.append(f"- `{name}` — {count}")
    lines.append("")
    lines.append("## Top 20 departments")
    for name, count in meta["top_departments"]:
        lines.append(f"- `{name}` — {count}")
    lines.append("")
    lines.append(
        f"Dedup rule: within **{DEDUP_DISTANCE_KM} km** and "
        f"**{DEDUP_TIME_DAYS} days**, same municipality when known."
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        required=True,
        help="A normalized JSONL file. Pass --input multiple times to merge sources.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/inventory/02_final"),
        help="Directory where the CSV / meta / report go.",
    )
    parser.add_argument(
        "--inventory-version",
        default="v1",
        help="Inventory version suffix (default: %(default)s).",
    )
    args = parser.parse_args(argv)

    inputs = [path for path in args.input if path.exists()]
    missing = [path for path in args.input if not path.exists()]
    if missing:
        for path in missing:
            print(f"Input not found: {path}", file=sys.stderr)
        if not inputs:
            return 1

    events: list[dict] = []
    source_counts: Counter = Counter()
    for path in inputs:
        loaded = _load_jsonl(path)
        events.extend(loaded)
        for event in loaded:
            source_counts[event.get("source") or "unknown"] += 1

    print(f"Loaded {len(events)} raw events from {len(inputs)} file(s).")

    clusters = _cluster(events)
    merged = [_merge_cluster(cluster) for cluster in clusters]
    merged.sort(
        key=lambda event: (
            event.get("observed_at") or "",
            event.get("municipality") or "",
            event.get("event_id") or "",
        )
    )
    print(
        f"Produced {len(merged)} deduplicated events "
        f"(collapsed {len(events) - len(merged)} duplicates)."
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / (
        f"colombia_landslide_events_{args.inventory_version}.csv"
    )
    meta_path = args.out_dir / (
        f"colombia_landslide_events_{args.inventory_version}.meta.json"
    )
    report_path = args.out_dir / (
        f"colombia_landslide_events_{args.inventory_version}.report.md"
    )

    _write_csv(csv_path, merged)
    meta = _build_meta(merged, source_counts)
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    report_path.write_text(_build_report(meta, inputs), encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {meta_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
