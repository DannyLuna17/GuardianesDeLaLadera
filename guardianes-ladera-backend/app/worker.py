from __future__ import annotations

import logging
import signal
import time

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.bootstrap import init_database, seed_demo_data
from app.db.session import session_scope
from app.services.structural_catalog import ensure_real_data_structural_catalog
from app.tasks.scheduler import BackendScheduler

logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.validate_real_data_runtime()
    init_database()
    if settings.seed_demo_data:
        with session_scope() as session:
            seed_demo_data(session)
    with session_scope() as session:
        ensure_real_data_structural_catalog(session, for_api=False)

    scheduler = BackendScheduler()
    if settings.enable_scheduler:
        scheduler.start()
        logger.info(
            "Worker scheduler started",
            extra={
                "execution_mode": settings.scheduler_execution_mode,
                "sources": settings.scheduler_sources,
                "pipeline_interval_minutes": settings.operational_pipeline_interval_minutes,
            },
        )
    else:
        logger.info("Worker started with scheduler disabled")

    should_stop = False

    def stop_handler(*_: object) -> None:
        nonlocal should_stop
        should_stop = True

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    while not should_stop:
        time.sleep(0.5)

    scheduler.shutdown()
    logger.info("Worker stopped")


if __name__ == "__main__":
    main()
