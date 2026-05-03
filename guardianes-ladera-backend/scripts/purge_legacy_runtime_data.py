from __future__ import annotations

from sqlalchemy import delete

from app.db.session import session_scope
from app.models import (
    HistoricalEvent,
    MunicipalityRainPoint,
    PredictionRun,
    RainOverlay,
    SourceSyncEvent,
    SourceSyncStatus,
    UngrdRecord,
    ZoneExplanation,
    ZonePrediction,
)


def main() -> None:
    with session_scope() as session:
        for model in (
            ZoneExplanation,
            ZonePrediction,
            PredictionRun,
            MunicipalityRainPoint,
            RainOverlay,
            HistoricalEvent,
            UngrdRecord,
            SourceSyncEvent,
            SourceSyncStatus,
        ):
            session.execute(delete(model))

    print("Purged legacy runtime data tables.")


if __name__ == "__main__":
    main()
