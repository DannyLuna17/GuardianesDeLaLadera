"""Normalize the UNGRD emergency CSV into our canonical landslide event schema.

Input: the public CSV from
    https://www.datos.gov.co/Ambiente-y-Desarrollo-Sostenible/Emergencias-UNGRD-/wwkg-r6te
encoded as UTF-8. A handful of rows contain stray bytes that are not valid
UTF-8; we read with ``errors='replace'`` so the ingest does not abort on them.

Output: JSONL — one canonical event per line — that the downstream
``merge_inventory.py`` script will combine with SIMMA (and anything else) before
writing the final master CSV the backend importer consumes.

Only events whose ``EVENTO`` column is ``MOVIMIENTO EN MASA`` or
``AVENIDA TORRENCIAL`` are kept (with case-insensitive matching). Avenidas
torrenciales are debris flows which are landslide-adjacent and decidedly in scope
for the Mocoa 2017 / 2022 Colombian landslide record.

UNGRD does NOT provide coordinates in the CSV — only municipality and department —
so every UNGRD record is tagged ``record_quality="medium"``. Precise geocoding
happens later during merge when we join against SIMMA or a municipality centroid
gazetteer.

Usage:

    uv run python scripts/landslide_inventory/normalize_ungrd.py \\
        --input ../data-raw/Emergencias_UNGRD._20260419.csv \\
        --output data/inventory/01_staging/ungrd_normalized.jsonl

"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


LANDSLIDE_EVENT_TYPES = {
    "MOVIMIENTO EN MASA",
    "AVENIDA TORRENCIAL",
}

DATE_FORMAT = "%Y %b %d %I:%M:%S %p"


def _parse_date(raw: str) -> datetime | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.strptime(raw, DATE_FORMAT)
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc)


def _parse_int(raw: str) -> int | None:
    raw = (raw or "").strip().replace(",", "")
    if not raw or raw in {"-", "NA", "N/A"}:
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def _classify_severity(
    *,
    deaths: int | None,
    missing: int | None,
    homes_destroyed: int | None,
    homes_damaged: int | None,
) -> tuple[str, str]:
    """Return ``(severity, record_quality)``.

    Follows the researcher's explicit hierarchy:
        fatal  -> any deaths
        severe -> any missing OR >=5 homes destroyed (proxy for displacement)
        moderate -> any home damaged or destroyed
        minor  -> all signals explicitly zero (small reported event)
    If none of the damage fields parsed to a number (all ``None``), apply the
    project's default-rule: ``severity="moderate"`` with ``record_quality="low"``
    so downstream consumers know the row is a best-effort guess, not data.
    """
    any_signal = (
        deaths is not None
        or missing is not None
        or homes_destroyed is not None
        or homes_damaged is not None
    )
    d = deaths or 0
    m = missing or 0
    dest = homes_destroyed or 0
    dmg = homes_damaged or 0
    if d > 0:
        return "fatal", "medium"
    if m > 0 or dest >= 5:
        return "severe", "medium"
    if dest > 0 or dmg > 0:
        return "moderate", "medium"
    if any_signal:
        return "minor", "medium"
    return "moderate", "low"


def _movement_type(evento: str) -> str:
    """Reduce UNGRD's coarse ``EVENTO`` taxonomy to our five-class scheme.

    ``AVENIDA TORRENCIAL`` is unambiguously a flow (debris / water-laden). For
    ``MOVIMIENTO EN MASA`` UNGRD does not publish a subtype and the landslide
    literature treats slides as the majority of mass-movement events; the
    researcher's brief also recommends ``slide`` as the safe default until the
    row's description text says otherwise. Downstream, the richer ``CLAS_MAPA``
    field on matched SIMMA clusters can refine this.
    """
    token = (evento or "").strip().upper()
    if token == "AVENIDA TORRENCIAL":
        return "flow"
    return "slide"


def _stable_event_id(
    *,
    observed_at: datetime,
    municipality: str,
    department: str,
    evento: str,
    divipola: str,
    row_index: int,
) -> str:
    # Hash covers (date, location, type); row_index disambiguates same-day
    # same-municipality duplicates within the CSV (UNGRD occasionally records
    # separate verrugas of one event).
    key = "|".join(
        [
            observed_at.date().isoformat(),
            (divipola or "").strip(),
            (municipality or "").strip().upper(),
            (department or "").strip().upper(),
            (evento or "").strip().upper(),
            str(row_index),
        ]
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return f"UNGRD-{digest}"


def normalize_file(
    *,
    input_path: Path,
    output_path: Path,
    encoding: str = "utf-8",
    encoding_errors: str = "replace",
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "total_rows": 0,
        "kept_rows": 0,
        "dropped_no_date": 0,
        "dropped_non_landslide": 0,
        "by_event_type": {},
        "by_year": {},
        "by_severity": {},
        "by_record_quality": {},
    }

    with (
        open(
            input_path,
            "r",
            encoding=encoding,
            errors=encoding_errors,
            newline="",
        ) as src,
        open(output_path, "w", encoding="utf-8") as dst,
    ):
        reader = csv.reader(src)
        header = next(reader)
        col = {name.strip(): idx for idx, name in enumerate(header)}

        def c(name: str) -> int | None:
            return col.get(name)

        idx_fecha = c("FECHA")
        idx_depto = c("DEPARTAMENTO")
        idx_muni = c("MUNICIPIO")
        idx_evento = c("EVENTO")
        idx_divipola = c("DIVIPOLA")
        idx_deaths = c("FALLECIDOS")
        idx_injured = c("HERIDOS")
        idx_missing = c("DESAPARECIDOS")
        idx_homes_dest = c("VIVIENDAS DESTRUIDAS")
        idx_homes_dmg = c("VIVIENDAS AVERIADAS")
        required = (
            idx_fecha,
            idx_depto,
            idx_muni,
            idx_evento,
            idx_divipola,
        )
        if any(value is None for value in required):
            raise RuntimeError(
                "UNGRD CSV is missing required columns "
                "(FECHA, DEPARTAMENTO, MUNICIPIO, EVENTO, DIVIPOLA)."
            )

        for row_index, row in enumerate(reader):
            stats["total_rows"] += 1
            try:
                evento_raw = row[idx_evento]
            except IndexError:
                continue
            evento_key = evento_raw.strip().upper()
            if evento_key not in LANDSLIDE_EVENT_TYPES:
                stats["dropped_non_landslide"] += 1
                continue
            observed_at = _parse_date(row[idx_fecha])
            if observed_at is None:
                stats["dropped_no_date"] += 1
                continue

            deaths = _parse_int(row[idx_deaths]) if idx_deaths is not None else None
            injured = (
                _parse_int(row[idx_injured]) if idx_injured is not None else None
            )
            missing = (
                _parse_int(row[idx_missing]) if idx_missing is not None else None
            )
            homes_dest = (
                _parse_int(row[idx_homes_dest])
                if idx_homes_dest is not None
                else None
            )
            homes_dmg = (
                _parse_int(row[idx_homes_dmg])
                if idx_homes_dmg is not None
                else None
            )
            homes_damaged = None
            if homes_dest is not None or homes_dmg is not None:
                homes_damaged = (homes_dest or 0) + (homes_dmg or 0)

            municipality = row[idx_muni].strip()
            department = row[idx_depto].strip()
            divipola = row[idx_divipola].strip().replace(",", "")

            severity, record_quality = _classify_severity(
                deaths=deaths,
                missing=missing,
                homes_destroyed=homes_dest,
                homes_damaged=homes_dmg,
            )

            event = {
                "event_id": _stable_event_id(
                    observed_at=observed_at,
                    municipality=municipality,
                    department=department,
                    evento=evento_key,
                    divipola=divipola,
                    row_index=row_index,
                ),
                "source": "UNGRD",
                "observed_at": observed_at.isoformat(),
                "municipality": municipality,
                "department": department,
                "divipola": divipola or None,
                "latitude": None,
                "longitude": None,
                "severity": severity,
                "movement_type": _movement_type(evento_raw),
                "deaths": deaths,
                "injured": injured,
                "missing": missing,
                "homes_destroyed": homes_dest,
                "homes_damaged": homes_damaged,
                "description": evento_raw.strip(),
                "source_url": (
                    "https://www.datos.gov.co/Ambiente-y-Desarrollo-Sostenible/"
                    "Emergencias-UNGRD-/wwkg-r6te"
                ),
                "record_quality": record_quality,
                "raw_row_index": row_index,
            }
            dst.write(json.dumps(event, ensure_ascii=False) + "\n")
            stats["kept_rows"] += 1
            stats["by_event_type"][evento_key] = (
                stats["by_event_type"].get(evento_key, 0) + 1
            )
            year_key = str(observed_at.year)
            stats["by_year"][year_key] = stats["by_year"].get(year_key, 0) + 1
            stats["by_severity"][severity] = (
                stats["by_severity"].get(severity, 0) + 1
            )
            stats["by_record_quality"][record_quality] = (
                stats["by_record_quality"].get(record_quality, 0) + 1
            )

    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the UNGRD emergencies CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/inventory/01_staging/ungrd_normalized.jsonl"),
        help="Where to write the normalized JSONL.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Encoding of the UNGRD CSV (default: %(default)s).",
    )
    parser.add_argument(
        "--encoding-errors",
        default="replace",
        help="What to do with invalid bytes (default: %(default)s).",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 1

    stats = normalize_file(
        input_path=args.input,
        output_path=args.output,
        encoding=args.encoding,
        encoding_errors=args.encoding_errors,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
