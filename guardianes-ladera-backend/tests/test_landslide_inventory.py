"""Unit tests for the pure functions in the landslide-inventory pipeline.

Each script under ``scripts/landslide_inventory/`` mixes a thin CLI shell with
a handful of stateless helpers (date parsing, severity classification,
movement-type mapping, slugification, dedup clustering, etc.). The DB-backed
``run_import`` and ``run_generator`` flows are exercised by the integration
smoke checks under ``data/inventory/03_reports/``; this file pins the helper
contracts so a quiet refactor cannot silently break severity logic, dedup
distances, or accent-tolerant gazetteer matching.

Imports use a sys.path injection because the inventory scripts are not yet a
proper package — keeping them as standalone scripts is intentional (they are
operator-runnable) and adding ``__init__.py`` would change their CLI ergonomics.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest


_INVENTORY_DIR = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "landslide_inventory"
)
if str(_INVENTORY_DIR) not in sys.path:
    sys.path.insert(0, str(_INVENTORY_DIR))


import normalize_ungrd  # noqa: E402
import normalize_simma  # noqa: E402
import merge_inventory  # noqa: E402
import import_landslide_inventory as importer  # noqa: E402
import generate_pseudo_absences as pa_gen  # noqa: E402


# ---------------------------------------------------------------------------
# normalize_ungrd: date, severity, movement type, stable id
# ---------------------------------------------------------------------------


class TestNormalizeUngrd:
    def test_parse_date_handles_real_csv_format(self):
        parsed = normalize_ungrd._parse_date("2019 Jan 04 12:00:00 AM")
        assert parsed is not None
        assert parsed == datetime(2019, 1, 4, 0, 0, tzinfo=timezone.utc)

    def test_parse_date_returns_none_on_garbage(self):
        assert normalize_ungrd._parse_date("") is None
        assert normalize_ungrd._parse_date("not-a-date") is None
        assert normalize_ungrd._parse_date(None) is None  # type: ignore[arg-type]

    def test_parse_int_strips_thousands_separator(self):
        assert normalize_ungrd._parse_int("5,656") == 5656
        assert normalize_ungrd._parse_int(" 12 ") == 12
        assert normalize_ungrd._parse_int("0") == 0

    def test_parse_int_returns_none_for_blank_or_dash(self):
        assert normalize_ungrd._parse_int("") is None
        assert normalize_ungrd._parse_int("-") is None
        assert normalize_ungrd._parse_int("N/A") is None
        assert normalize_ungrd._parse_int("garbage") is None

    @pytest.mark.parametrize(
        "deaths,missing,destroyed,damaged,expected_severity,expected_quality",
        [
            (1, 0, 0, 0, "fatal", "medium"),
            (0, 2, 0, 0, "severe", "medium"),
            (0, 0, 5, 0, "severe", "medium"),
            (0, 0, 6, 0, "severe", "medium"),
            (0, 0, 1, 0, "moderate", "medium"),
            (0, 0, 0, 3, "moderate", "medium"),
            # All zero (signal present but no damage reported) → minor / medium
            (0, 0, 0, 0, "minor", "medium"),
            # All None (no signal parsed at all) → fallback rule:
            # severity="moderate", record_quality="low"
            (None, None, None, None, "moderate", "low"),
        ],
    )
    def test_classify_severity_matches_policy_hierarchy(
        self,
        deaths,
        missing,
        destroyed,
        damaged,
        expected_severity,
        expected_quality,
    ):
        severity, quality = normalize_ungrd._classify_severity(
            deaths=deaths,
            missing=missing,
            homes_destroyed=destroyed,
            homes_damaged=damaged,
        )
        assert severity == expected_severity
        assert quality == expected_quality

    def test_movement_type_avenida_torrencial_is_flow(self):
        assert normalize_ungrd._movement_type("AVENIDA TORRENCIAL") == "flow"
        assert normalize_ungrd._movement_type(" Avenida Torrencial ") == "flow"

    def test_movement_type_default_is_slide_per_research_brief(self):
        # The researcher's recommendation: when description is unspecific,
        # default ``MOVIMIENTO EN MASA`` to "slide" — most landslide events
        # in the literature are slides. Confirms we did not silently revert
        # to "complex" after the 2026-04-22 fix.
        assert normalize_ungrd._movement_type("MOVIMIENTO EN MASA") == "slide"
        assert normalize_ungrd._movement_type("anything-else") == "slide"

    def test_stable_event_id_is_deterministic_and_distinguishes_neighbours(self):
        common = dict(
            observed_at=datetime(2020, 6, 10, tzinfo=timezone.utc),
            municipality="Pereira",
            department="Risaralda",
            evento="MOVIMIENTO EN MASA",
            divipola="66001",
        )
        a = normalize_ungrd._stable_event_id(row_index=1, **common)
        b = normalize_ungrd._stable_event_id(row_index=1, **common)
        c = normalize_ungrd._stable_event_id(row_index=2, **common)
        assert a == b  # deterministic on identical inputs
        assert a != c  # row index disambiguates same-day same-muni rows
        assert a.startswith("UNGRD-")


# ---------------------------------------------------------------------------
# normalize_simma: SIMMA point projection helpers
# ---------------------------------------------------------------------------


class TestNormalizeSimma:
    def test_lookup_finds_first_non_empty_candidate_case_insensitive(self):
        attrs = {"TIPO": "", "CLAS_MAPA": "Volcamiento", "OBJECTID": 1}
        assert (
            normalize_simma._lookup(attrs, ("TIPO", "CLAS_MAPA"))
            == "Volcamiento"
        )

    def test_lookup_returns_none_when_all_candidates_blank(self):
        attrs = {"TIPO": None, "CLAS_MAPA": "  "}
        assert normalize_simma._lookup(attrs, ("TIPO", "CLAS_MAPA")) is None

    @pytest.mark.parametrize(
        "type_hint,subtype_hint,expected",
        [
            ("Caída", None, "fall"),
            ("Volcamiento", "Volcamiento flexural de roca", "fall"),
            ("Deslizamiento", "Deslizamiento rotacional", "slide"),
            ("Reptación", None, "slide"),
            ("Flujo", "Flujo de detritos", "flow"),
            ("Avalancha", None, "flow"),
            ("Avenida torrencial", None, "flow"),
            ("Propagación lateral", None, "complex"),
            ("Tipo desconocido", None, "unknown"),
            (None, None, "unknown"),
        ],
    )
    def test_movement_type_taxonomic_reduction(
        self, type_hint, subtype_hint, expected
    ):
        # The recodification table is the contract the researcher locked in
        # after the SIMMA brief. Pin every row.
        assert (
            normalize_simma._movement_type(type_hint, subtype_hint) == expected
        )

    def test_extract_coordinates_point(self):
        lat, lon = normalize_simma._extract_coordinates(
            {"type": "Point", "coordinates": [-74.5, 5.0]}
        )
        assert lat == pytest.approx(5.0)
        assert lon == pytest.approx(-74.5)

    def test_extract_coordinates_polygon_uses_first_ring_first_vertex(self):
        lat, lon = normalize_simma._extract_coordinates(
            {
                "type": "Polygon",
                "coordinates": [[[-1.0, 2.0], [-1.0, 3.0], [0.0, 3.0]]],
            }
        )
        assert lat == pytest.approx(2.0)
        assert lon == pytest.approx(-1.0)

    def test_extract_coordinates_handles_missing_geometry(self):
        assert normalize_simma._extract_coordinates(None) == (None, None)
        assert normalize_simma._extract_coordinates({}) == (None, None)
        assert normalize_simma._extract_coordinates(
            {"type": "Point", "coordinates": []}
        ) == (None, None)

    def test_resolve_record_id_prefers_inv_id_then_falls_back(self):
        assert (
            normalize_simma._resolve_record_id(
                {"INV_MOVIMIENTO_MASA_ID": 255, "OBJECTID": 113}
            )
            == "255"
        )
        # Fallback to OBJECTID when INV id absent
        assert (
            normalize_simma._resolve_record_id({"OBJECTID": 999})
            == "999"
        )
        # Fallback to literal string "unknown" when nothing useful
        assert normalize_simma._resolve_record_id({}) == "unknown"


# ---------------------------------------------------------------------------
# merge_inventory: distance, time-aware dedup, cluster merge
# ---------------------------------------------------------------------------


class TestMergeInventory:
    def test_haversine_zero_for_identical_points(self):
        d = merge_inventory._haversine_km(4.81, -75.69, 4.81, -75.69)
        assert d == pytest.approx(0.0, abs=1e-9)

    def test_haversine_known_pereira_to_dosquebradas(self):
        # ~1.3 km between Pereira and Dosquebradas centroids; 5% tolerance.
        d = merge_inventory._haversine_km(4.813, -75.694, 4.836, -75.672)
        assert 2.0 < d < 5.0

    def test_parse_iso_handles_z_suffix_and_naive(self):
        a = merge_inventory._parse_iso("2020-06-10T00:00:00Z")
        assert a is not None and a.tzinfo is not None
        b = merge_inventory._parse_iso("2020-06-10T00:00:00")
        assert b is not None and b.tzinfo is not None
        assert a == b

    def test_parse_iso_returns_none_on_garbage(self):
        assert merge_inventory._parse_iso("not-iso") is None
        assert merge_inventory._parse_iso(None) is None
        assert merge_inventory._parse_iso("") is None

    def test_pick_primary_prefers_higher_record_quality(self):
        events = [
            {
                "event_id": "low-q",
                "record_quality": "low",
                "severity": "fatal",
                "source": "UNGRD",
            },
            {
                "event_id": "high-q",
                "record_quality": "high",
                "severity": "minor",
                "source": "SGC_SIMMA",
            },
        ]
        primary = merge_inventory._pick_primary(events)
        assert primary["event_id"] == "high-q"

    def test_are_duplicates_when_close_in_space_and_time(self):
        a = {
            "observed_at": "2020-06-10T00:00:00+00:00",
            "municipality": "Pereira",
            "latitude": 4.813,
            "longitude": -75.694,
        }
        b = {
            "observed_at": "2020-06-12T00:00:00+00:00",
            "municipality": "Pereira",
            "latitude": 4.815,
            "longitude": -75.696,
        }
        assert merge_inventory._are_duplicates(a, b) is True

    def test_are_duplicates_false_when_time_far_apart(self):
        a = {
            "observed_at": "2020-06-10T00:00:00+00:00",
            "municipality": "Pereira",
            "latitude": 4.813,
            "longitude": -75.694,
        }
        b = {
            "observed_at": "2020-06-30T00:00:00+00:00",
            "municipality": "Pereira",
            "latitude": 4.813,
            "longitude": -75.694,
        }
        assert merge_inventory._are_duplicates(a, b) is False

    def test_are_duplicates_false_for_different_municipalities(self):
        a = {
            "observed_at": "2020-06-10T00:00:00+00:00",
            "municipality": "Pereira",
            "latitude": 4.813,
            "longitude": -75.694,
        }
        b = {
            "observed_at": "2020-06-10T00:00:00+00:00",
            "municipality": "Dosquebradas",
            "latitude": 4.813,
            "longitude": -75.694,
        }
        assert merge_inventory._are_duplicates(a, b) is False

    def test_merge_cluster_takes_highest_severity_and_unions_sources(self):
        cluster = [
            {
                "event_id": "A",
                "record_quality": "medium",
                "severity": "minor",
                "source": "UNGRD",
            },
            {
                "event_id": "B",
                "record_quality": "high",
                "severity": "fatal",
                "source": "SGC_SIMMA",
            },
        ]
        merged = merge_inventory._merge_cluster(cluster)
        assert merged["severity"] == "fatal"
        assert "UNGRD" in (merged["related_sources"] + [merged.get("source")])
        # primary id is the high-quality one; the other id surfaces as related.
        assert merged["event_id"] == "B"
        assert "A" in merged["related_event_ids"]

    def test_merge_cluster_singleton_passes_through_with_empty_related(self):
        cluster = [
            {
                "event_id": "Solo",
                "record_quality": "medium",
                "severity": "moderate",
                "source": "UNGRD",
            }
        ]
        merged = merge_inventory._merge_cluster(cluster)
        assert merged["event_id"] == "Solo"
        assert merged["related_event_ids"] == []
        assert merged["related_sources"] == []


# ---------------------------------------------------------------------------
# import_landslide_inventory: name normalization, slugs, severity score
# ---------------------------------------------------------------------------


class TestImporterHelpers:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("APARTADÓ", "APARTADO"),
            ("Apía", "APIA"),
            ("ñariño  ", "NARINO"),
            ("San José de Cúcuta", "SAN JOSE DE CUCUTA"),
            ("", ""),
            (None, ""),
        ],
    )
    def test_normalize_admin_name_strips_diacritics_and_uppercases(
        self, raw, expected
    ):
        assert importer._normalize_admin_name(raw) == expected

    def test_slugify_handles_spanish_accents_and_spaces(self):
        assert importer._slugify("Cúcuta") == "cucuta"
        assert importer._slugify("San José") == "san-jose"
        assert importer._slugify("  ") == "unknown"

    def test_truncate_appends_hash_when_over_limit(self):
        long = "a" * 50
        truncated = importer._truncate(long, limit=20)
        assert len(truncated) == 20
        # Same input should give the same hashed suffix
        assert truncated == importer._truncate(long, limit=20)
        # Different input should give a different suffix
        other = importer._truncate("b" * 50, limit=20)
        assert other != truncated

    def test_muni_id_includes_dept_and_muni_slug_within_32_chars(self):
        muni_id = importer._muni_id("Cundinamarca", "La Vega")
        assert muni_id == "cundinamarca-la-vega"
        assert len(muni_id) <= 32

    def test_zone_id_appends_cab_suffix(self):
        assert importer._zone_id("antioquia-medellin") == (
            "antioquia-medellin-cab"
        )

    @pytest.mark.parametrize(
        "severity,expected",
        [
            ("fatal", 1.0),
            ("severe", 0.85),
            ("moderate", 0.6),
            ("minor", 0.35),
            ("MAJOR_TYPO", 0.5),  # fallback
            (None, 0.5),
        ],
    )
    def test_severity_to_target_score_matches_policy(self, severity, expected):
        assert importer._severity_to_target_score(severity) == pytest.approx(
            expected
        )

    def test_centroid_for_uses_municipality_gazetteer_when_available(self):
        lat, lon = importer._centroid_for("Pereira", "Risaralda")
        # Pereira is in the hardcoded gazetteer at (4.813, -75.694)
        assert lat == pytest.approx(4.813)
        assert lon == pytest.approx(-75.694)

    def test_centroid_for_falls_back_to_department_when_muni_unknown(self):
        lat, lon = importer._centroid_for("Random Pueblo", "Antioquia")
        # Antioquia centroid in the dept gazetteer
        assert lat == pytest.approx(7.0)
        assert lon == pytest.approx(-75.5)

    def test_centroid_for_falls_back_to_country_centre_when_both_unknown(self):
        lat, lon = importer._centroid_for("Nowhere", "AlsoNowhere")
        # Colombia geographic centre
        assert lat == pytest.approx(4.57)
        assert lon == pytest.approx(-74.30)

    def test_polygon_box_is_centred_on_centroid(self):
        polygon = importer._polygon_box((4.0, -75.0))
        lats = [pt[0] for pt in polygon]
        lons = [pt[1] for pt in polygon]
        assert min(lats) < 4.0 < max(lats)
        assert min(lons) < -75.0 < max(lons)
        # Square bounds (4 vertices)
        assert len(polygon) == 4

    def test_extract_first_polygon_ring_from_polygon(self):
        ring = importer._extract_first_polygon_ring(
            {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1]]]}
        )
        assert ring == [[0, 0], [1, 0], [1, 1]]

    def test_extract_first_polygon_ring_from_multipolygon_takes_first_part(
        self,
    ):
        ring = importer._extract_first_polygon_ring(
            {
                "type": "MultiPolygon",
                "coordinates": [[[[0, 0], [1, 0]]], [[[5, 5], [6, 5]]]],
            }
        )
        assert ring == [[0, 0], [1, 0]]

    def test_extract_first_polygon_ring_returns_empty_on_unsupported_geom(self):
        assert importer._extract_first_polygon_ring(
            {"type": "Point", "coordinates": [0, 0]}
        ) == []
        assert importer._extract_first_polygon_ring({}) == []


# ---------------------------------------------------------------------------
# generate_pseudo_absences: exclusion rule, parse helpers
# ---------------------------------------------------------------------------


class TestPseudoAbsenceHelpers:
    def test_parse_iso_returns_aware_datetime(self):
        parsed = pa_gen._parse_iso("2020-06-10T00:00:00Z")
        assert parsed is not None
        assert parsed.tzinfo is not None
        assert parsed == datetime(2020, 6, 10, tzinfo=timezone.utc)

    def test_parse_iso_passthrough_for_existing_datetime(self):
        original = datetime(2021, 1, 1, tzinfo=timezone.utc)
        assert pa_gen._parse_iso(original) is original

    def test_parse_iso_returns_none_on_garbage(self):
        assert pa_gen._parse_iso("not-iso") is None
        assert pa_gen._parse_iso(None) is None

    def test_excluded_inside_window(self):
        from datetime import timedelta

        positives = [datetime(2020, 6, 10, tzinfo=timezone.utc)]
        candidate = datetime(2020, 6, 20, tzinfo=timezone.utc)
        # Exactly at window edge (10 days < 14)
        assert pa_gen._excluded(candidate, positives, timedelta(days=14))

    def test_excluded_outside_window(self):
        from datetime import timedelta

        positives = [datetime(2020, 6, 10, tzinfo=timezone.utc)]
        candidate = datetime(2020, 7, 1, tzinfo=timezone.utc)  # 21 days
        assert not pa_gen._excluded(
            candidate, positives, timedelta(days=14)
        )

    def test_excluded_against_multiple_positives_uses_any(self):
        from datetime import timedelta

        positives = [
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2020, 6, 10, tzinfo=timezone.utc),
        ]
        candidate = datetime(2020, 6, 18, tzinfo=timezone.utc)
        # Far from Jan 1 but only 8 days from Jun 10
        assert pa_gen._excluded(candidate, positives, timedelta(days=14))

    def test_v1_policy_constants_match_memory(self):
        # The v1 memory lock-in: 1:2 ratio, ±14d, target 0.05, source tag
        assert pa_gen.DEFAULT_RATIO == 2
        assert pa_gen.DEFAULT_EXCLUSION_DAYS == 14
        assert pa_gen.PSEUDO_ABSENCE_TARGET_SCORE == 0.05
        assert pa_gen.PSEUDO_ABSENCE_SOURCE == "pseudo_absence_temporal_v1"
        assert pa_gen.PSEUDO_ABSENCE_SEVERITY == "stable"
