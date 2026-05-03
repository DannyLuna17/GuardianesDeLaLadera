# Landslide inventory pipeline

This folder assembles a canonical Colombian landslide inventory from public
sources (UNGRD, SGC SIMMA, and future additions) into a single CSV that the
backend's label-import flow can ingest.

## Directory layout this pipeline produces

```
data/inventory/
├── 00_raw/
│   ├── ungrd/Emergencias_UNGRD._YYYYMMDD.csv
│   └── simma/
│       ├── lotes/lote_00001_ids_00000001_to_00001000.geojson
│       ├── lotes/lote_00002_...
│       └── simma_clasificacion.json
├── 01_staging/
│   ├── ungrd_normalized.jsonl
│   └── simma_normalized.jsonl
└── 02_final/
    ├── colombia_landslide_events_v1.csv
    ├── colombia_landslide_events_v1.meta.json
    └── colombia_landslide_events_v1.report.md
```

The CSV in `02_final/` is the handoff to the backend: every row carries
`event_id`, `source`, `observed_at`, `municipality`, `department`, `latitude`,
`longitude`, `severity`, `movement_type`, and damage fields, plus
`record_quality` (`high` | `medium` | `low`) and the ids/sources of any records
that were deduplicated into it.

## Scripts

All scripts live under `guardianes-ladera-backend/scripts/landslide_inventory/`
and are intended to be run with `uv run python <script>` from the backend dir.

### `fetch_simma.py` — download SGC SIMMA

Downloads the SIMMA mass-movement inventory via its public ArcGIS REST service
and, by default, also downloads the 1,123 Colombian municipal polygons plus 33
departmental polygons from SIMMA `Capas_Generales` (layers 5 and 6). The
admin-boundary polygons are what lets `normalize_simma.py` spatial-join each
landslide point to its municipality.

Resumable: individual GeoJSON chunk files are only re-fetched if missing. POST
is used for queries with long `objectIds` lists (GET hits ArcGIS's URL-length
cap at ~1000 ids).

```bash
uv run python scripts/landslide_inventory/fetch_simma.py \
    --output-dir data/inventory/00_raw/simma

# Only the points (skip admin boundaries) — smaller footprint
uv run python scripts/landslide_inventory/fetch_simma.py \
    --output-dir data/inventory/00_raw/simma \
    --skip-admin-boundaries

# smoke test against one chunk of points only
uv run python scripts/landslide_inventory/fetch_simma.py \
    --output-dir data/inventory/00_raw/simma \
    --max-chunks 1 --skip-admin-boundaries
```

SIMMA occasionally reorders layers or renames fields. If SGC changes the base
URL, point the script at the new one with `--base-url`. Layer ids
(`--points-layer`, `--classification-layer`, `--municipios-layer`,
`--departamentos-layer`) are also configurable; visit
`https://srvags.sgc.gov.co/arcgis/rest/services/SIMMA/Capas_Principales/MapServer?f=json`
and
`https://srvags.sgc.gov.co/arcgis/rest/services/SIMMA/Capas_Generales/MapServer?f=json`
in a browser to enumerate layers when something breaks.

### `normalize_ungrd.py` — convert UNGRD CSV to canonical JSONL

Filters the UNGRD emergencies CSV (latin-1) to `MOVIMIENTO EN MASA` and
`AVENIDA TORRENCIAL` events and emits one canonical event per line. UNGRD does
NOT carry lat/lon in the public CSV — every UNGRD record is tagged
`record_quality="medium"` and will be geocoded later during merge against
SIMMA or a municipality-centroid gazetteer.

```bash
uv run python scripts/landslide_inventory/normalize_ungrd.py \
    --input ../data-raw/Emergencias_UNGRD._20260419.csv \
    --output data/inventory/01_staging/ungrd_normalized.jsonl
```

### `normalize_simma.py` — convert SIMMA chunks to canonical JSONL

Reads every `lote_*.geojson` under `--lotes-dir` and projects its attributes
into the same schema as `normalize_ungrd.py`. When `--municipios` is provided
(default: `data/inventory/00_raw/simma/boundaries/municipios.geojson` from
`fetch_simma.py`), each SIMMA point is **spatial-joined to its Colombian
municipality**: the script builds an in-memory bbox-prefilter + ray-casting
index over the 1,123 `Capas_Generales/Municipios` polygons and populates the
record's `municipality`, `department`, and `divipola` (= `COD_DEPART` +
`COD_MUNICI`) fields. The 9,177-point inventory spatial-joins in ~12 seconds.

**Reality check**: the SIMMA public MapServer still exposes only spatial
geometry and movement-type taxonomy — no dates, no severities. The spatial
join fills municipality/department **but does not add `observed_at`**. Every
SIMMA record therefore carries `severity = null` and either
`record_quality = "medium"` (when the spatial join matched, i.e. point + muni
+ dept) or `record_quality = "spatial_prior_only"` (when it did not; for the
current inventory that is 5/9,177 offshore points near Buenaventura). Either
way these records are **not outcome labels** — the backend's label schema
requires `observed_at`. They are excellent as spatial priors for (a) zone
catalog creation with real admin polygons, (b) cross-referencing UNGRD
records whose municipality matches, (c) pseudo-absence exclusion.

**Movement-type mapping** (matches the SGC taxonomy from the researcher's
brief):
- `Caída / Caída de rocas / Volcamiento (flexural)` → `fall`
- `Deslizamiento / rotacional / traslacional / por licuación / Reptación` → `slide`
- `Flujo / Flujo de detritos / Flujo de lodo / Avalancha / Avenida torrencial` → `flow`
- `Propagación Lateral / Deformación gravitacional profunda / Complejo` → `complex`
- anything else → `unknown`

```bash
uv run python scripts/landslide_inventory/normalize_simma.py \
    --lotes-dir data/inventory/00_raw/simma/lotes \
    --output data/inventory/01_staging/simma_normalized.jsonl
```

### `merge_inventory.py` — dedup and ship the master CSV

Accepts any number of normalized JSONL inputs, clusters events within a
**3 km / 3 day** window (same municipality, when known), keeps the best-quality
representative per cluster, and writes:

- `colombia_landslide_events_v1.csv` — one row per unique event
- `colombia_landslide_events_v1.meta.json` — counts by source, quality, year, top municipalities/departments
- `colombia_landslide_events_v1.report.md` — human-readable summary

```bash
uv run python scripts/landslide_inventory/merge_inventory.py \
    --input data/inventory/01_staging/ungrd_normalized.jsonl \
    --input data/inventory/01_staging/simma_normalized.jsonl \
    --out-dir data/inventory/02_final
```

### `import_landslide_inventory.py` — push labels into the backend DB

Reads `colombia_landslide_events_v1.csv` and writes, inside one transaction:

- `Municipality` rows for every new municipality (with coarse centroid from a
  built-in gazetteer of the most-mentioned municipalities, with department-
  centroid fallback, with Colombia-centre fallback).
- One default `Zone` per municipality (id = `<muni-slug>-cab`,
  type = `Cabecera municipal`, polygon = small box around centroid).
- `ZoneOutcomeLabel` rows keyed by `(zone_id, observed_at, source)` — the
  uniqueness constraint the domain model enforces. Severity is mapped to
  `target_score` as: fatal=1.0, severe=0.85, moderate=0.60, minor=0.35.

SIMMA rows (`record_quality == "spatial_prior_only"`) are **skipped** — they
have no `observed_at` and cannot be outcome labels. They remain valuable as a
spatial prior for a future zone-refinement pass.

The script is idempotent: re-runs report the existing rows as duplicates and
insert nothing.

```bash
# Dry-run (validate without committing)
uv run python scripts/landslide_inventory/import_landslide_inventory.py \
    --csv data/inventory/02_final/colombia_landslide_events_v1.csv \
    --dry-run

# Real import
uv run python scripts/landslide_inventory/import_landslide_inventory.py \
    --csv data/inventory/02_final/colombia_landslide_events_v1.csv
```

Smoke-checked 2026-04-20: fresh SQLite DB ingest produced 637 municipalities,
637 zones, 3,222 labels across 32 Colombian departments. Top zones by label
count: Pereira 33, San Francisco (Cundinamarca) 30, La Vega 28, Villavicencio
25, Toledo 25. Second run: 0 new inserts, 3,222 duplicates correctly skipped.

## Full pipeline (from a clean state)

```bash
# 1. UNGRD was already downloaded manually into ../data-raw/.
#    Just normalize it.
uv run python scripts/landslide_inventory/normalize_ungrd.py \
    --input ../data-raw/Emergencias_UNGRD._20260419.csv \
    --output data/inventory/01_staging/ungrd_normalized.jsonl

# 2. Pull SIMMA. Can take hours the first time; skipped on re-runs.
uv run python scripts/landslide_inventory/fetch_simma.py \
    --output-dir data/inventory/00_raw/simma

# 3. Normalize SIMMA.
uv run python scripts/landslide_inventory/normalize_simma.py \
    --lotes-dir data/inventory/00_raw/simma/lotes \
    --output data/inventory/01_staging/simma_normalized.jsonl

# 4. Merge all sources into the canonical inventory.
uv run python scripts/landslide_inventory/merge_inventory.py \
    --input data/inventory/01_staging/ungrd_normalized.jsonl \
    --input data/inventory/01_staging/simma_normalized.jsonl \
    --out-dir data/inventory/02_final

# 5. Import the inventory into the backend DB.
uv run python scripts/landslide_inventory/import_landslide_inventory.py \
    --csv data/inventory/02_final/colombia_landslide_events_v1.csv
```

After step 5 the database holds `Municipality` + `Zone` + `ZoneOutcomeLabel`
+ `HistoricalEvent` rows (the importer emits a mirrored `HistoricalEvent` for
every UNGRD label so `ZoneFeatureBuilder` sees real historical frequencies).

Step 6 seeds one backdated synthetic `PredictionRun` so labels can resolve a
prediction for their `observed_at`:

```bash
uv run python scripts/landslide_inventory/backfill_historical_predictions.py
```

This creates one `PredictionRun` (model_version = `synthetic_historical_backfill_v1`)
dated one day before the earliest label, plus one `ZonePrediction` +
`ZoneExplanation` per zone that has any labels. Feature snapshots are
computed lazily at export time by `ZoneFeatureBuilder` using whatever the DB
currently has.

**Step 6.5 — Pseudo-absences (policy `pseudo_absence_temporal_v1`)**.
Materialize the project's pre-decided pseudo-absence policy: per-zone
temporal negatives at 1:2 ratio with ±14d exclusion, target_score=0.05,
fixed RNG seed for reproducibility. Without this step the dataset is
positive-only and `beta_regression` cannot fit its precision parameter
cleanly:

```bash
uv run python scripts/landslide_inventory/generate_pseudo_absences.py
```

Smoke-checked 2026-04-27 on the fresh inventory DB: 637 zones with positives
→ 5,852 negatives inserted (95% of the 6,132 at 1:2 ratio); 168 single-day
zones skipped because ±14d exclusion left no temporal room. Idempotent — a
second run reports the existing rows as already seeded and inserts zero.

Step 7 runs the governed benchmark/review on real labels:

```bash
uv run python scripts/landslide_inventory/run_benchmark_on_real_labels.py \
    --output data/inventory/03_reports/benchmark_decision.md \
    --max-labels 500
```

Smoke-run 2026-04-22 on the fresh inventory DB produced a real ranking under
`temporal_holdout_backtest` + `nested_outer_estimate`: **xgboost 1st**
(val RMSE 0.072), linear_ridge 5th, additive_spline 8th, beta_regression
14th. Promotion is (correctly) blocked by the stability-window gate —
`minimum_nested_outer_selection_rate_not_met` — because a single-snapshot
win isn't enough evidence under the research-aligned policy.

Re-run 2026-04-27 with pseudo-absences included (9,074-row dataset)
flipped the champion to **additive_spline 1st** (val RMSE 0.0158, narrowly
ahead of xgboost at 0.0159), confirming the v1 stability conclusion. All
RMSEs collapsed ~4× because the bimodal target distribution makes
dominant-class prediction trivial. `beta_regression` finally fit its
precision parameter and its absolute RMSE dropped 6× (0.17 → 0.027) even
though its rank stayed at 14. Reports are at
`data/inventory/03_reports/benchmark_with_pseudo_absences.md`.

### Feature-sparsity caveat

The step-7 benchmark trains on real UNGRD outcome labels but its feature
snapshots cover only what the DB currently carries: `HistoricalEvent` counts
per municipality and zone (real) plus road-segment intersections and rain
overlays for the seeded Mocoa/Pasto/Popayán zones (real). The 637
auto-created municipal zones carry **zero values** for rain, slope, geology,
and road features because IDEAM historical rainfall, SGC topography, and OSM
roads have not been ingested yet. The champion pick is therefore a
**historical-frequency prior**, not a definitive decision. Expect the
ranking to tighten (and potentially flip toward the interpretable families)
once those layers are joined in. The feature-ingestion research prompt
sitting with the researcher is the canonical unblock for that.

## Policy reminders (already decided)

- **Scope**: every Colombian municipality where we can accumulate ≥5 qualifying events survives. Do not filter to Mocoa/Pasto/Popayán.
- **Pseudo-absences**: temporal `v1` — 1:2 positive-to-negative, ±14d exclusion, target `0.05`, source `pseudo_absence_temporal_v1`. Generated by a separate script that runs after labels are ingested — not inside this inventory pipeline.
- **Severity taxonomy**: `fatal` > `severe` > `moderate` > `minor`. See the per-source normalizer for the exact mapping rules.
