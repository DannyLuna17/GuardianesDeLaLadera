"""Read IDEAM daily-precipitation ZIP and expose per-station rolling features.

Input: ``PrecipitacionNacionalDiaria.zip`` from
    ``http://bart.ideam.gov.co/PQRS/AQTSUtils/PrecipitacionNacionalDiaria.zip``
plus the cached station catalog produced by ``fetch_ideam_catalog.py``.

What this module provides:

- ``IdeamPrecipitationStore``: an on-disk index over the ZIP that loads any
  station's daily series on demand and caches the parsed result. Memory cost
  is bounded by ``--cache-max-stations`` (default 64).
- ``rolling_window_features(date, station_series)``: given a target date and
  a station's daily series, returns the canonical feature dictionary the
  backend's ZoneFeatureBuilder consumes — ``rain_24h``, ``rain_72h``,
  ``rain_7d``, ``rain_14d``, ``rain_30d`` accumulations in millimetres, plus
  ``rain_observed_days_in_window`` so feature consumers can detect when a
  station has gaps inside a window.
- ``find_nearest_station(catalog, lat, lon)``: simple nearest-neighbour lookup
  by haversine distance.

Used as a module by the per-zone feature builder, but also runnable as a CLI
to dump one station's series or one zone's per-label rainfall features for
inspection.

Usage:

    # CLI: dump one station's daily series as a CSV
    uv run python scripts/feature_ingestion/normalize_ideam_precipitation.py \\
        --zip ../data-raw/PrecipitacionNacionalDiaria.zip \\
        --catalog data/feature_ingestion/00_raw/ideam_station_catalog.csv \\
        --station 26135040 \\
        --output data/feature_ingestion/01_staging/ideam_pereira_2022.csv \\
        --start 2022-01-01 --end 2022-12-31

    # CLI: dump rolling-window features for a station on a single date
    uv run python scripts/feature_ingestion/normalize_ideam_precipitation.py \\
        --zip ../data-raw/PrecipitacionNacionalDiaria.zip \\
        --catalog data/feature_ingestion/00_raw/ideam_station_catalog.csv \\
        --station 26135040 --rolling-as-of 2020-06-15

"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import zipfile
from collections import OrderedDict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable


_SCRIPT_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _SCRIPT_DIR.parent.parent
_SCRIPTS_DIR = _SCRIPT_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _lib.geo import haversine_km as _haversine_km  # noqa: E402


ZIP_FILENAME_PREFIX = "PTPM_CON_INTER@"
ZIP_FILENAME_SUFFIX = ".data"
DEFAULT_WINDOWS_DAYS = (1, 3, 7, 14, 30)
DEFAULT_CACHE_MAX_STATIONS = 64


def _data_filename(station_id: str | int) -> str:
    return f"{ZIP_FILENAME_PREFIX}{station_id}{ZIP_FILENAME_SUFFIX}"


def parse_station_series(raw: bytes) -> list[tuple[date, float]]:
    """Parse one station's ``Fecha|Valor`` daily file.

    Empty lines, the header, and rows with non-numeric values are silently
    skipped. ``observed_at`` is normalized to a ``date`` (no tz, no time):
    IDEAM records each value at 07:00 local time but the value itself is the
    24-hour accumulation for that calendar day.
    """
    text = raw.decode("utf-8", errors="replace")
    series: list[tuple[date, float]] = []
    for line_no, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line or line_no == 0:
            continue
        if "|" not in line:
            continue
        fecha_raw, valor_raw = line.split("|", 1)
        try:
            observed_at = datetime.strptime(
                fecha_raw.strip(), "%Y-%m-%d %H:%M:%S"
            ).date()
        except ValueError:
            continue
        try:
            value = float(valor_raw.strip())
        except ValueError:
            continue
        if math.isnan(value):
            continue
        series.append((observed_at, value))
    return series


class IdeamPrecipitationStore:
    """Lazy reader over the IDEAM ZIP with bounded LRU station cache."""

    def __init__(
        self,
        zip_path: Path,
        *,
        cache_max_stations: int = DEFAULT_CACHE_MAX_STATIONS,
    ) -> None:
        if not zip_path.exists():
            raise FileNotFoundError(zip_path)
        self.zip_path = zip_path
        self._zip = zipfile.ZipFile(zip_path)
        self._available_ids = {
            name[len(ZIP_FILENAME_PREFIX) : -len(ZIP_FILENAME_SUFFIX)]
            for name in self._zip.namelist()
            if name.startswith(ZIP_FILENAME_PREFIX)
            and name.endswith(ZIP_FILENAME_SUFFIX)
        }
        self._cache: OrderedDict[str, list[tuple[date, float]]] = OrderedDict()
        self._cache_max_stations = cache_max_stations

    def __enter__(self) -> "IdeamPrecipitationStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        try:
            self._zip.close()
        except Exception:  # noqa: BLE001 — closing must not raise
            pass

    def has_station(self, station_id: str | int) -> bool:
        return str(station_id) in self._available_ids

    def available_ids(self) -> set[str]:
        return set(self._available_ids)

    def series_for(self, station_id: str | int) -> list[tuple[date, float]]:
        sid = str(station_id)
        cached = self._cache.get(sid)
        if cached is not None:
            self._cache.move_to_end(sid)
            return cached
        if sid not in self._available_ids:
            return []
        with self._zip.open(_data_filename(sid)) as src:
            raw = src.read()
        parsed = parse_station_series(raw)
        self._cache[sid] = parsed
        self._cache.move_to_end(sid)
        if len(self._cache) > self._cache_max_stations:
            self._cache.popitem(last=False)
        return parsed


def load_catalog(catalog_path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(catalog_path, "r", encoding="utf-8", newline="") as src:
        for row in csv.DictReader(src):
            try:
                lat = float(row["latitud"])
                lon = float(row["longitud"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(
                {
                    "idestacion": row["idestacion"],
                    "nombre": row.get("nombre") or "",
                    "idcategoria": row.get("idcategoria") or "",
                    "latitud": lat,
                    "longitud": lon,
                }
            )
    return rows


def find_nearest_station(
    catalog: list[dict],
    lat: float,
    lon: float,
    *,
    require_in_zip: set[str] | None = None,
    max_km: float | None = None,
) -> dict | None:
    """Return the catalog row of the nearest station to ``(lat, lon)``."""
    candidates = catalog
    if require_in_zip is not None:
        candidates = [
            row for row in catalog if row["idestacion"] in require_in_zip
        ]
    if not candidates:
        return None
    best: dict | None = None
    best_km = float("inf")
    for row in candidates:
        d = _haversine_km(lat, lon, row["latitud"], row["longitud"])
        if d < best_km:
            best_km = d
            best = dict(row, distance_km=round(d, 4))
    if best is None or (max_km is not None and best_km > max_km):
        return None
    return best


def rolling_window_features(
    target_date: date,
    series: list[tuple[date, float]],
    *,
    windows_days: Iterable[int] = DEFAULT_WINDOWS_DAYS,
) -> dict[str, float | int]:
    """Compute rolling rainfall accumulations ending at ``target_date``.

    The conventions are: a window of ``N`` days ends on ``target_date``
    inclusive (so ``rain_1d`` is just that day's rainfall, ``rain_3d`` is the
    sum of ``target_date``, ``target_date - 1``, ``target_date - 2``). This
    matches how the literature defines antecedent rainfall windows (e.g.
    Han & Semnani 2025) and how the existing ``ZoneFeatureBuilder`` consumes
    them.
    """
    if not series:
        return {f"rain_{n}d": 0.0 for n in windows_days} | {
            f"rain_{n}d_observed_days": 0 for n in windows_days
        }
    # Build a date->mm dict for O(1) lookup
    by_date = {d: v for d, v in series}
    out: dict[str, float | int] = {}
    for n in windows_days:
        total = 0.0
        observed = 0
        for offset in range(n):
            d = target_date - timedelta(days=offset)
            value = by_date.get(d)
            if value is not None:
                total += value
                observed += 1
        out[f"rain_{n}d"] = round(total, 3)
        out[f"rain_{n}d_observed_days"] = observed
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip",
        type=Path,
        required=True,
        help="Path to PrecipitacionNacionalDiaria.zip.",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path(
            "data/feature_ingestion/00_raw/ideam_station_catalog.csv"
        ),
    )
    parser.add_argument(
        "--station",
        required=True,
        help="IDEAM idestacion to dump.",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Inclusive start date YYYY-MM-DD (omit for full series).",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Inclusive end date YYYY-MM-DD (omit for full series).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="If set, write the daily series CSV here.",
    )
    parser.add_argument(
        "--rolling-as-of",
        default=None,
        help=(
            "Optional YYYY-MM-DD. If set, also print rolling rainfall"
            " features ending at that date."
        ),
    )
    args = parser.parse_args(argv)

    catalog = load_catalog(args.catalog)
    catalog_by_id = {row["idestacion"]: row for row in catalog}
    catalog_row = catalog_by_id.get(str(args.station))
    if catalog_row is None:
        print(
            f"Station {args.station} not in catalog {args.catalog}",
            file=sys.stderr,
        )
        return 1

    with IdeamPrecipitationStore(args.zip) as store:
        series = store.series_for(args.station)
        if not series:
            print(
                f"Station {args.station} has no daily series in {args.zip}",
                file=sys.stderr,
            )
            return 1
        start = (
            datetime.strptime(args.start, "%Y-%m-%d").date()
            if args.start
            else series[0][0]
        )
        end = (
            datetime.strptime(args.end, "%Y-%m-%d").date()
            if args.end
            else series[-1][0]
        )
        filtered = [(d, v) for d, v in series if start <= d <= end]

        summary = {
            "station": args.station,
            "name": catalog_row["nombre"],
            "category": catalog_row["idcategoria"],
            "lat": catalog_row["latitud"],
            "lon": catalog_row["longitud"],
            "rows_in_zip_total": len(series),
            "rows_in_window": len(filtered),
            "first_in_window": filtered[0][0].isoformat() if filtered else None,
            "last_in_window": filtered[-1][0].isoformat() if filtered else None,
        }
        if args.rolling_as_of:
            target = datetime.strptime(args.rolling_as_of, "%Y-%m-%d").date()
            summary["rolling_features"] = rolling_window_features(
                target, series
            )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8", newline="") as dst:
                writer = csv.writer(dst)
                writer.writerow(
                    [
                        "station_id",
                        "name",
                        "latitude",
                        "longitude",
                        "date",
                        "precipitation_mm",
                        "source",
                    ]
                )
                for d, v in filtered:
                    writer.writerow(
                        [
                            args.station,
                            catalog_row["nombre"],
                            catalog_row["latitud"],
                            catalog_row["longitud"],
                            d.isoformat(),
                            v,
                            "IDEAM",
                        ]
                    )
            summary["output_path"] = str(args.output)

    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
