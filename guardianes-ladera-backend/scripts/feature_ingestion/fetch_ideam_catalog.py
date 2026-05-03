"""Fetch and cache the IDEAM national station catalog.

The catalog at the IDEAM ArcGIS REST endpoint exposes every active hydro-met
station with its identifier (``idestacion``), name, category, and lat/lon. The
project already integrates this endpoint per municipality via the existing
provider adapter — this script captures the **full national catalog** in one
file so downstream feature builders can find the nearest station for any
zone without re-hitting the network on every run.

Output: a single CSV at ``data/feature_ingestion/00_raw/ideam_station_catalog.csv``
with one row per active precipitation station.

Schema:
    idestacion, nombre, idcategoria, latitud, longitud, source

The script paginates the ArcGIS service in chunks of 1000 records to stay well
inside the public-service rate limits, and is resumable: rerunning it skips
the network call when the cached CSV already exists and is non-empty (pass
``--force`` to refetch).

Usage:

    uv run python scripts/feature_ingestion/fetch_ideam_catalog.py

"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from urllib.parse import urlencode


_SCRIPT_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _SCRIPT_DIR.parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))


CATALOG_BASE_URL = (
    "https://visualizador.ideam.gov.co/gisserver/rest/services/CNE/"
    "CatalogoNacionalEstaciones/MapServer/0/query"
)
PRECIPITATION_CATEGORIES = (
    "AM", "CO", "CP", "ME", "PG", "PM", "SP", "SS",
)
ACTIVE_STATE = "ESTA001"
PAGE_SIZE = 1000
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 90.0


def _build_query(*, offset: int) -> dict:
    where = (
        f"idestadoestaciontm='{ACTIVE_STATE}' AND idcategoria IN "
        + "(" + ",".join(f"'{c}'" for c in PRECIPITATION_CATEGORIES) + ")"
    )
    return {
        "f": "json",
        "where": where,
        "outFields": "idestacion,nombre,idcategoria,latitud,longitud",
        "returnGeometry": "false",
        "orderByFields": "idestacion ASC",
        "resultOffset": str(offset),
        "resultRecordCount": str(PAGE_SIZE),
    }


def _fetch_page(client, *, offset: int) -> list[dict]:
    query = _build_query(offset=offset)
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.get(
                CATALOG_BASE_URL,
                params=query,
                timeout=DEFAULT_TIMEOUT,
            )
            response.raise_for_status()
            features = response.json().get("features") or []
            return [feature.get("attributes") or {} for feature in features]
        except Exception as exc:  # noqa: BLE001 — retry every transport hiccup
            last_exc = exc
            if attempt == MAX_RETRIES:
                break
            sleep_for = 2.0 ** attempt
            print(
                f"  [retry {attempt}/{MAX_RETRIES}] "
                f"{type(exc).__name__}: {exc!s} — sleeping {sleep_for:.1f}s",
                file=sys.stderr,
            )
            time.sleep(sleep_for)
    assert last_exc is not None
    raise last_exc


def fetch_catalog(*, output_path: Path, force: bool = False) -> dict:
    import httpx  # local import: keeps script importable in env without httpx

    if output_path.exists() and output_path.stat().st_size > 0 and not force:
        print(
            f"Catalog already cached at {output_path} "
            f"({output_path.stat().st_size} bytes); pass --force to refetch."
        )
        with open(output_path, "r", encoding="utf-8", newline="") as src:
            reader = csv.DictReader(src)
            cached_rows = sum(1 for _ in reader)
        return {
            "rows": cached_rows,
            "from_cache": True,
            "output_path": str(output_path),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    seen_ids: set[str] = set()
    offset = 0

    with httpx.Client() as client:
        while True:
            print(
                f"Fetching catalog rows {offset:>6} … {offset + PAGE_SIZE:>6}"
            )
            page = _fetch_page(client, offset=offset)
            if not page:
                break
            new_rows = [
                {
                    "idestacion": str(item.get("idestacion") or "").strip(),
                    "nombre": str(item.get("nombre") or "").strip(),
                    "idcategoria": str(item.get("idcategoria") or "").strip(),
                    "latitud": item.get("latitud"),
                    "longitud": item.get("longitud"),
                    "source": "IDEAM",
                }
                for item in page
            ]
            for row in new_rows:
                if (
                    row["idestacion"]
                    and row["latitud"] is not None
                    and row["longitud"] is not None
                    and row["idestacion"] not in seen_ids
                ):
                    rows.append(row)
                    seen_ids.add(row["idestacion"])
            if len(page) < PAGE_SIZE:
                break
            offset += PAGE_SIZE

    with open(output_path, "w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(
            dst,
            fieldnames=[
                "idestacion",
                "nombre",
                "idcategoria",
                "latitud",
                "longitud",
                "source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} stations to {output_path}")
    return {
        "rows": len(rows),
        "from_cache": False,
        "output_path": str(output_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/feature_ingestion/00_raw/ideam_station_catalog.csv"
        ),
        help="Where to write the cached station catalog CSV.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refetch even if the cached file already exists.",
    )
    args = parser.parse_args(argv)
    summary = fetch_catalog(output_path=args.output, force=args.force)
    import json

    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
