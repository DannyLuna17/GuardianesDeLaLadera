# Feature ingestion pipeline

Ingesters that fetch and normalize the **dynamic and structural feature
sources** the model needs for real-data scoring. Sister pipeline to
`scripts/landslide_inventory/` (which produces labels). Each source lives in
its own pair of scripts: a fetcher (downloads / caches raw artefacts) and a
normalizer (projects them onto the canonical feature schema the backend
consumes).

The scope and verification of each source are documented in the research
brief checked into the project. As of this writing only the IDEAM
precipitation pair has shipped; the other sources are scheduled in priority
order: CHIRPS fallback, Geofabrik OSM roads, SGC Capas_Generales boundaries,
SGC geology, IGAC soils, Hansen forest loss, DEM-derived slope.

---

## IDEAM — daily precipitation history

**Status**: shipped 2026-04-28.

### Sources

- **Station catalog** — public ArcGIS REST endpoint
  `https://visualizador.ideam.gov.co/gisserver/rest/services/CNE/CatalogoNacionalEstaciones/MapServer/0/query`,
  paginated 1000 records per request. No auth required.
- **Daily precipitation series** — public ZIP at
  `http://bart.ideam.gov.co/PQRS/AQTSUtils/PrecipitacionNacionalDiaria.zip`
  (~138 MB). One pipe-separated `.data` file per station, schema
  `Fecha|Valor` with daily totals back to as early as 1947 for some stations
  (Pereira Matecaña has 28,328 daily records 1947-09-01 through 2026-02-22).
  Verified live by the project owner; the URL is published in the official
  IDEAM PQRSDF guidance.

### Scripts

#### `fetch_ideam_catalog.py`

Caches the national station catalog (active precipitation stations only) as
a single CSV. Run once; subsequent runs reuse the cache unless `--force` is
passed. Filters to active stations (`idestadoestaciontm = 'ESTA001'`) and
the precipitation categories AM/CO/CP/ME/PG/PM/SP/SS.

```bash
uv run python scripts/feature_ingestion/fetch_ideam_catalog.py
```

Smoke-checked 2026-04-28: 3,663 active stations cached at
`data/feature_ingestion/00_raw/ideam_station_catalog.csv` (~5 paginated
requests, completes in seconds).

#### `normalize_ideam_precipitation.py`

Reader + feature aggregator over the precipitation ZIP. Importable as a
module (`IdeamPrecipitationStore`, `find_nearest_station`,
`rolling_window_features`) or runnable as a CLI to dump one station's daily
series and the rolling-window features at any target date.

CLI smoke check:

```bash
uv run python scripts/feature_ingestion/normalize_ideam_precipitation.py \
    --zip ../data-raw/PrecipitacionNacionalDiaria.zip \
    --station 26135040 \
    --start 2022-01-01 --end 2022-12-31 \
    --output data/feature_ingestion/01_staging/ideam_pereira_2022.csv \
    --rolling-as-of 2022-05-15
```

Reproduced for Pereira Aeropuerto Matecaña (id `26135040`):

- 365 daily records exported for the 2022 calendar year (full coverage).
- Rolling features at 2022-05-15: `rain_1d=0.0`, `rain_3d=34.2`,
  `rain_7d=101.8`, `rain_14d=142.7`, `rain_30d=378.0` mm.
  Each window also reports `rain_Nd_observed_days` so feature consumers can
  detect station gaps rather than silently treating missing as zero.

### Module API

The main module exports four pure functions and one class. Use them from a
downstream feature backfill instead of re-running the CLI per zone:

```python
from normalize_ideam_precipitation import (
    IdeamPrecipitationStore,
    load_catalog,
    find_nearest_station,
    rolling_window_features,
)

catalog = load_catalog(Path(".../ideam_station_catalog.csv"))

with IdeamPrecipitationStore(Path(".../data-raw/PrecipitacionNacionalDiaria.zip")) as store:
    station = find_nearest_station(
        catalog,
        zone.centroid_lat, zone.centroid_lon,
        require_in_zip=store.available_ids(),
        max_km=50.0,
    )
    if station is None:
        # zone is too far from any IDEAM station with daily data
        ...
    series = store.series_for(station["idestacion"])
    features = rolling_window_features(label.observed_at.date(), series)
    # features → {"rain_1d": ..., "rain_3d": ..., ...}
```

### Convention

- All distances use haversine in kilometres. WGS84 (EPSG:4326) latitude /
  longitude.
- Rolling windows are inclusive of the target date and extend backward
  N days. `rain_3d` on 2022-05-15 = sum of the 13th, 14th, and 15th.
- Default windows: 1, 3, 7, 14, 30 days. Match the antecedent-rainfall
  windows used in the landslide-susceptibility literature.
- Target date with no observation still aggregates history correctly — the
  observation-count surface in the output is the operator's signal for "the
  station has gaps inside this window".

### Testing

20 unit tests in `tests/test_feature_ingestion_ideam.py` lock the parser,
distance, nearest-station and rolling-window contracts. Run:

```bash
uv run pytest tests/test_feature_ingestion_ideam.py -q
```

The full ZIP-driven integration is exercised by the smoke check above; do
not pin it as an automated test (140 MB binary is too heavy for CI).

---

## Pending sources

The remaining feature sources are queued in the priority order recommended
by the research brief:

1. **CHIRPS v2** — fallback to IDEAM for any zone whose nearest station
   exceeds the `max_km` threshold or has too many gaps in the antecedent
   window. Public NetCDF/COG/ERDDAP, no auth.
2. **OSM road network** — bulk via Geofabrik Colombia (`colombia-latest-free.gpkg.zip`),
   with Overpass reserved for spot validation.
3. **SGC geology** — `Mapa_Geologico_Colombia_V2023` MapServer, public REST.
4. **IGAC soils** — `agrologia/suelosdecolombiaaniveldeorden` MapServer
   (`Correlacion_Suelos_Ordenes_100k`).
5. **Hansen Global Forest Change** — public GCS mirror, no GEE.
6. **DEM** — pending choice between Copernicus GLO-30 / SRTM v3 /
   ASTER GDEM v3. Decide before writing ingester.

Each source will follow the same shape as IDEAM: a fetcher that caches the
raw artefact, a normalizer that exposes the canonical feature accessors,
and a downstream backfill that joins the features to the imported labels at
their `observed_at` dates.
