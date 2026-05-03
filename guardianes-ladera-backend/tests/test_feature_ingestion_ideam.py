"""Unit tests for the IDEAM precipitation primitives.

These pin the contracts of every pure function in
``scripts/feature_ingestion/normalize_ideam_precipitation.py``: the daily
``Fecha|Valor`` parser, the haversine distance, the nearest-station lookup
with optional ZIP-availability filter, and the rolling-window rainfall
feature aggregator. The full ZIP-driven integration (140 MB file, 4,446
stations) is exercised by the smoke check in the project README; this file
locks the helper-level contracts so a quiet refactor cannot drift them.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest


_INVENTORY_DIR = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "feature_ingestion"
)
if str(_INVENTORY_DIR) not in sys.path:
    sys.path.insert(0, str(_INVENTORY_DIR))


import normalize_ideam_precipitation as ideam  # noqa: E402


# ---------------------------------------------------------------------------
# parse_station_series
# ---------------------------------------------------------------------------


class TestParseStationSeries:
    def test_parses_pipe_separated_daily_records(self):
        raw = (
            "Fecha|Valor\n"
            "2022-05-13 07:00:00|12.4\n"
            "2022-05-14 07:00:00|0.0\n"
            "2022-05-15 07:00:00|34.7\n"
        ).encode("utf-8")
        series = ideam.parse_station_series(raw)
        assert series == [
            (date(2022, 5, 13), 12.4),
            (date(2022, 5, 14), 0.0),
            (date(2022, 5, 15), 34.7),
        ]

    def test_skips_header_blank_lines_and_unparseable_rows(self):
        raw = (
            "Fecha|Valor\n"
            "\n"
            "2022-05-13 07:00:00|12.4\n"
            "garbage line without separator\n"
            "2022-05-14 07:00:00|not-a-number\n"
            "bad-date 99:99:99|2.0\n"
            "2022-05-15 07:00:00|0.0\n"
        ).encode("utf-8")
        series = ideam.parse_station_series(raw)
        assert series == [
            (date(2022, 5, 13), 12.4),
            (date(2022, 5, 15), 0.0),
        ]

    def test_drops_nan_values(self):
        raw = (
            "Fecha|Valor\n"
            "2022-01-01 07:00:00|nan\n"
            "2022-01-02 07:00:00|2.0\n"
        ).encode("utf-8")
        series = ideam.parse_station_series(raw)
        assert series == [(date(2022, 1, 2), 2.0)]

    def test_handles_invalid_utf8_gracefully(self):
        # UNGRD encoding is UTF-8 but with stray bytes; IDEAM is plain UTF-8 in
        # practice, but the parser should not crash on stray bytes either.
        raw = b"Fecha|Valor\n2022-01-01 07:00:00|3.5\n\xff\xfe garbage\n"
        series = ideam.parse_station_series(raw)
        assert series == [(date(2022, 1, 1), 3.5)]

    def test_empty_file(self):
        assert ideam.parse_station_series(b"") == []
        assert ideam.parse_station_series(b"Fecha|Valor\n") == []


# ---------------------------------------------------------------------------
# _haversine_km
# ---------------------------------------------------------------------------


class TestHaversine:
    def test_zero_distance_for_identical_points(self):
        d = ideam._haversine_km(4.81, -75.69, 4.81, -75.69)
        assert d == pytest.approx(0.0, abs=1e-9)

    def test_known_pereira_to_dosquebradas(self):
        # ~2.4 km between Pereira and Dosquebradas centroids.
        d = ideam._haversine_km(4.813, -75.694, 4.836, -75.672)
        assert 2.0 < d < 5.0

    def test_known_bogota_to_medellin(self):
        # Real distance is ~242 km; allow 5% tolerance.
        d = ideam._haversine_km(4.66, -74.08, 6.244, -75.574)
        assert 230 < d < 255


# ---------------------------------------------------------------------------
# find_nearest_station
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_catalog() -> list[dict]:
    return [
        {
            "idestacion": "26135040",
            "nombre": "AEROPUERTO MATECANA",
            "idcategoria": "SP",
            "latitud": 4.81586111,
            "longitud": -75.73722222,
        },
        {
            "idestacion": "26120450",
            "nombre": "CRIOLINDA",
            "idcategoria": "PM",
            "latitud": 4.68333333,
            "longitud": -75.58333333,
        },
        {
            "idestacion": "21205012",
            "nombre": "BOGOTA EL DORADO",
            "idcategoria": "SP",
            "latitud": 4.7016,
            "longitud": -74.1469,
        },
    ]


class TestFindNearestStation:
    def test_returns_pereira_matecana_for_pereira_centre(
        self, sample_catalog
    ):
        nearest = ideam.find_nearest_station(
            sample_catalog, 4.813, -75.694
        )
        assert nearest is not None
        assert nearest["idestacion"] == "26135040"
        assert nearest["distance_km"] < 5.0

    def test_returns_bogota_station_for_bogota_centre(
        self, sample_catalog
    ):
        nearest = ideam.find_nearest_station(
            sample_catalog, 4.66, -74.08
        )
        assert nearest is not None
        assert nearest["idestacion"] == "21205012"

    def test_returns_none_when_outside_max_km(
        self, sample_catalog
    ):
        # All sample stations are >50 km from the Caribbean coast.
        nearest = ideam.find_nearest_station(
            sample_catalog, 11.0, -74.8, max_km=20.0
        )
        assert nearest is None

    def test_filters_by_zip_availability_when_required(
        self, sample_catalog
    ):
        # Pretend only the Bogotá station has data in the ZIP.
        nearest = ideam.find_nearest_station(
            sample_catalog,
            4.813,
            -75.694,
            require_in_zip={"21205012"},
        )
        assert nearest is not None
        assert nearest["idestacion"] == "21205012"

    def test_returns_none_when_catalog_empty(self):
        assert ideam.find_nearest_station([], 4.813, -75.694) is None

    def test_returns_none_when_zip_filter_excludes_everything(
        self, sample_catalog
    ):
        nearest = ideam.find_nearest_station(
            sample_catalog,
            4.813,
            -75.694,
            require_in_zip=set(),
        )
        assert nearest is None


# ---------------------------------------------------------------------------
# rolling_window_features
# ---------------------------------------------------------------------------


class TestRollingWindowFeatures:
    def _series(self, *pairs: tuple[str, float]) -> list[tuple[date, float]]:
        return [(date.fromisoformat(d), v) for d, v in pairs]

    def test_empty_series_returns_zeroed_feature_dict(self):
        out = ideam.rolling_window_features(date(2022, 5, 15), [])
        assert out == {
            "rain_1d": 0.0,
            "rain_3d": 0.0,
            "rain_7d": 0.0,
            "rain_14d": 0.0,
            "rain_30d": 0.0,
            "rain_1d_observed_days": 0,
            "rain_3d_observed_days": 0,
            "rain_7d_observed_days": 0,
            "rain_14d_observed_days": 0,
            "rain_30d_observed_days": 0,
        }

    def test_single_day_window_returns_only_that_day(self):
        series = self._series(
            ("2022-05-13", 5.0),
            ("2022-05-14", 8.0),
            ("2022-05-15", 12.0),
        )
        out = ideam.rolling_window_features(
            date(2022, 5, 15), series, windows_days=(1,)
        )
        assert out["rain_1d"] == pytest.approx(12.0)
        assert out["rain_1d_observed_days"] == 1

    def test_three_day_window_sums_inclusive_of_target(self):
        series = self._series(
            ("2022-05-13", 5.0),
            ("2022-05-14", 8.0),
            ("2022-05-15", 12.0),
        )
        out = ideam.rolling_window_features(
            date(2022, 5, 15), series, windows_days=(3,)
        )
        assert out["rain_3d"] == pytest.approx(25.0)  # 5 + 8 + 12
        assert out["rain_3d_observed_days"] == 3

    def test_window_with_gap_counts_only_observed_days(self):
        # No record for 2022-05-14 — the window aggregator should report
        # rain_3d_observed_days = 2, not 3, so feature consumers can detect
        # the gap rather than treating missing as zero.
        series = self._series(
            ("2022-05-13", 5.0),
            ("2022-05-15", 12.0),
        )
        out = ideam.rolling_window_features(
            date(2022, 5, 15), series, windows_days=(3,)
        )
        assert out["rain_3d"] == pytest.approx(17.0)
        assert out["rain_3d_observed_days"] == 2

    def test_target_date_with_no_record_still_aggregates_history(self):
        series = self._series(
            ("2022-05-13", 4.0),
            ("2022-05-14", 6.0),
            # No record for 2022-05-15 itself
        )
        out = ideam.rolling_window_features(
            date(2022, 5, 15), series, windows_days=(3,)
        )
        assert out["rain_3d"] == pytest.approx(10.0)
        assert out["rain_3d_observed_days"] == 2

    def test_default_windows_match_literature_convention(self):
        # The default set is 1, 3, 7, 14, 30 — common antecedent-rainfall
        # windows in landslide-susceptibility literature. Pin this so a
        # well-meaning refactor cannot drop one and break feature parity.
        assert ideam.DEFAULT_WINDOWS_DAYS == (1, 3, 7, 14, 30)
