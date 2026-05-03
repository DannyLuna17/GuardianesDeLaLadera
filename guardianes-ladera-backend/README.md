# Guardianes de la Ladera Backend

FastAPI backend for the Guardianes de la Ladera project.

This backend is designed around a predictive-model workflow for landslide-risk monitoring. The model is the decision engine, and the explanation layer comes after scoring. Today the explanation path is template-based, and the live LLM stage remains an optional final step controlled by configuration.

The runtime now defaults to `REAL_DATA_ONLY=true`. In that mode the backend refuses demo seeding, seed/stub adapters, and any structural catalog without official provenance.

The current pilot bundle is already checked into the repo at `app/data/official-structural/official_structural_bundle.json`. It was generated from official DANE and INVIAS exports for `Mocoa`, `Pasto`, and `Popayan`, and currently contains `3` municipalities, `12` official pilot-scope zones, and `7` INVIAS road segments.

## What This Backend Does

- serves the dashboard data contract used by the frontend
- ingests and normalizes source data for risk monitoring
- stores geospatial, temporal, operational, and governance data
- serves persisted prediction runs and their explanation history
- trains, evaluates, tunes, promotes, and rolls back predictive models
- manages governed outcome labels for supervised learning
- runs drift and shadow monitoring against labeled datasets
- opens operator review tasks from predictive monitoring alerts

## Technology Stack

- FastAPI
- SQLAlchemy 2.0
- Alembic
- PostgreSQL + PostGIS for containerized and production-oriented execution
- SQLite for fast local development fallback
- APScheduler for worker-side scheduled jobs
- PyJWT for auth
- GeoAlchemy2 for PostGIS geometry support

## Project Layout

```text
guardianes-ladera-backend/
  app/
    api/              HTTP routes and dependencies
    core/             settings, logging, security, exceptions
    data/             official raw exports, generated structural bundle, and legacy seed artifacts
    db/               session, bootstrap, migrations helpers, spatial helpers
    integrations/     HTTP provider adapters plus legacy seed adapters/parsers
    ml/               artifacts, datasets, inference, training, drift, shadow
    models/           SQLAlchemy domain models
    repositories/     read-focused repository layer
    schemas/          Pydantic request and response models
    services/         business and operational workflows
    tasks/            scheduler jobs
    main.py           API entrypoint
    worker.py         worker entrypoint
  alembic/            migration environment and revisions
  docs/               backend documentation set
  scripts/            operational helper scripts
  tests/              pytest suite
  Dockerfile
  docker-compose.yml
```

## Quick Start

### Local development

```powershell
uv sync
uv run python scripts/build_official_structural_bundle.py
uv run python scripts/import_official_structural_catalog.py
uv run uvicorn app.main:app --reload
```

With the default `REAL_DATA_ONLY=true` policy, ingestion requires valid provider base URLs, the app will not seed demo records on startup, and the API will refuse to boot until an official structural catalog bundle has been imported. The default bundle path is `app/data/official-structural/official_structural_bundle.json`.

The bundle-build script reads the raw official source archives already staged in `app/data/`:

- `MGN2025_MPIO_GRAFICO.zip`
- `MGN2025_URB_SECCION.zip`
- `shp_CRVeredas_2024.zip`
- `invias.geojson`

Those raw archives are local build inputs. The repository now ignores the large zip/xlsx source files plus temporary extracted inspection folders under `app/data/`; commit the generated official bundle, not the raw downloads.

If the bundle has already been generated and committed, you can skip the build step and import it directly.

Open:

- API: `http://127.0.0.1:8000`
- OpenAPI UI: `http://127.0.0.1:8000/docs`

### Worker

```powershell
uv run python -m app.worker
```

The worker can run with the scheduler disabled or enabled through environment variables.

### Tests

```powershell
uv run pytest -q
```

### Docker Compose

```powershell
docker compose up -d --build
```

This brings up:

- `postgres`
- `migrate`
- `runtime-bootstrap`
- `api`
- `worker`

`runtime-bootstrap` imports the official structural catalog, reruns official ingestion, and creates a fresh real-data run before `api` and `worker` are allowed to start. If the bundle is missing or invalid, Compose will now fail at `runtime-bootstrap` instead of crashing the API and worker in a loop.

`runtime-bootstrap` now logs each stage of the rebuild plus source-by-source ingestion progress. Provider HTTP calls default to `30s` timeouts with `3` retry attempts and `1.5s` backoff in the compose profile, so initial startup can pause for a few minutes while live providers respond.

## Default Local Credentials

If `REAL_DATA_ONLY=false` and seed data is enabled, the default admin account is:

- username: `admin`
- password: `guardianes-admin`

These values come from [.env.example](./.env.example) and should be changed for shared environments.

## Real Data Policy

- `REAL_DATA_ONLY=true` is the default local and compose setting.
- `SEED_DEMO_DATA` must stay disabled in that mode.
- Provider transports may use `auto` or `http`, but missing base URLs now fail clearly instead of falling back to seed data.
- Provider HTTP adapters retry transient timeout and URL errors with configurable `PROVIDER_REQUEST_TIMEOUT_SECONDS`, `PROVIDER_REQUEST_RETRY_ATTEMPTS`, and `PROVIDER_REQUEST_RETRY_BACKOFF_SECONDS`.
- Stub notification channels are not part of the default runtime configuration.
- The structural catalog now needs official provenance metadata on municipalities, zones, and road segments. Seeded geometry is rejected at startup and by runtime services.
- Build the structural catalog bundle with `scripts/build_official_structural_bundle.py` from official DANE and INVIAS exports, then import it with `scripts/import_official_structural_catalog.py`.
- The current checked-in bundle uses DANE `MGN 2025 nivel municipio`, DANE `MGN 2025 seccion urbana`, DANE `Veredas 2024`, and INVIAS `Red Vial`.
- Rebuild the full live runtime with `scripts/rebuild_official_runtime.py` once the bundle is in place.
- `/v1/admin/runs/trigger` now uses the operational real-data scoring path, but it will fail closed if the structural catalog has not been imported officially.

## Key Design Principle

The backend is intentionally model-first:

1. source data is ingested and normalized
2. spatial and temporal features are assembled
3. prediction runs are executed and stored
4. monitoring, drift, and challenger checks are performed
5. explanations are generated after model outputs already exist

That keeps the LLM out of the core risk decision path.
