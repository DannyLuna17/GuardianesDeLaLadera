from __future__ import annotations

import logging
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.bootstrap import init_database
from app.db.session import session_scope
from app.services.ingestion import IngestionService
from app.services.runs import RunService
from app.services.structural_catalog import (
    ensure_real_data_structural_catalog,
    import_structural_catalog_bundle,
    load_structural_catalog_bundle,
)

logger = logging.getLogger(__name__)


def resolve_bundle_path(argv: list[str]) -> Path:
    settings = get_settings()
    if len(argv) > 2:
        raise SystemExit(
            "Usage: uv run python scripts/rebuild_official_runtime.py [bundle.json]"
        )
    if len(argv) == 2:
        return Path(argv[1]).resolve()
    return settings.resolved_structural_catalog_bundle_path


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    bundle_path = resolve_bundle_path(sys.argv)
    logger.info("Starting official runtime rebuild from %s", bundle_path)
    bundle = load_structural_catalog_bundle(bundle_path)
    logger.info(
        "Loaded structural bundle: municipalities=%s zones=%s road_segments=%s",
        len(bundle.municipalities),
        len(bundle.zones),
        len(bundle.road_segments),
    )

    logger.info("Initializing database and migrations for runtime rebuild")
    init_database()
    with session_scope() as session:
        logger.info("Importing official structural catalog into runtime database")
        counts = import_structural_catalog_bundle(session, bundle)
        ensure_real_data_structural_catalog(session, for_api=False)
        logger.info(
            "Structural catalog import completed: municipalities=%s zones=%s road_segments=%s",
            counts["municipalities"],
            counts["zones"],
            counts["road_segments"],
        )

        logger.info(
            "Starting official ingestion cycle for sources=%s",
            ",".join(settings.scheduler_sources),
        )
        ingestion = IngestionService(session).sync_sources(
            source_ids=settings.scheduler_sources,
            origin="manual",
            note="Runtime rebuild after official structural catalog import.",
        )
        logger.info(
            "Official ingestion cycle completed: job_status=%s results=%s",
            ingestion.job.status,
            ", ".join(
                f"{item.source_id}:{item.status}:{item.processed_records}"
                for item in ingestion.synced_sources
            ),
        )
        logger.info("Starting first operational real-data scoring run")
        run = RunService(session).trigger_run(
            note="Official runtime rebuild run after structural catalog import."
        )
        logger.info(
            "Operational real-data scoring run completed: run_id=%s model_version=%s zones_monitored=%s",
            run.run.id,
            run.run.model_version,
            run.run.zones_monitored,
        )

    print(f"Rebuilt runtime from {bundle_path}")
    print(
        "Structural import:",
        f"{counts['municipalities']} municipalities,",
        f"{counts['zones']} zones,",
        f"{counts['road_segments']} road segments.",
    )
    print(
        "Ingestion:",
        ", ".join(
            f"{item.source_id}:{item.processed_records}"
            for item in ingestion.synced_sources
        ),
    )
    print(
        "Run:",
        f"id={run.run.id}",
        f"model={run.run.model_version}",
        f"zones={run.run.zones_monitored}",
    )


if __name__ == "__main__":
    main()
