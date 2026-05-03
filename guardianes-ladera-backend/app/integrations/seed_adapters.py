from __future__ import annotations

from sqlalchemy.orm import Session

from app.data.seed_store import load_seed_payload
from app.integrations.base import SyncResult
from app.integrations.normalizers import sync_historical_events, sync_rain_series, sync_ungrd_records


class BaseSeedAdapter:
    source_id: str
    transport = "seed"

    def __init__(self) -> None:
        self.seed = load_seed_payload()

    def sync(self, session: Session) -> SyncResult:
        raise NotImplementedError


class IdeamSeedAdapter(BaseSeedAdapter):
    source_id = "IDEAM"

    def sync(self, session: Session) -> SyncResult:
        total = sync_rain_series(session, self.seed["rainSeries"])
        return SyncResult(
            source_id=self.source_id,
            processed_records=total,
            status="completed",
            message="Seed-based IDEAM rain series synchronized.",
            adapter_key="seed.ideam",
            transport=self.transport,
            details={
                "dataset": "municipality_rain_points",
                "municipality_count": len(self.seed["rainSeries"]),
            },
        )


class SgcSeedAdapter(BaseSeedAdapter):
    source_id = "SGC"

    def sync(self, session: Session) -> SyncResult:
        total = sync_historical_events(session, self.seed["historicalEvents"])
        return SyncResult(
            source_id=self.source_id,
            processed_records=total,
            status="completed",
            message="Seed-based SGC historical events synchronized.",
            adapter_key="seed.sgc",
            transport=self.transport,
            details={
                "dataset": "historical_events",
                "record_type": "mass_movement_event",
            },
        )


class UngrdSeedAdapter(BaseSeedAdapter):
    source_id = "UNGRD"

    def sync(self, session: Session) -> SyncResult:
        total = sync_ungrd_records(session, self.seed["ungrdRecords"])
        return SyncResult(
            source_id=self.source_id,
            processed_records=total,
            status="completed",
            message="Seed-based UNGRD records synchronized.",
            adapter_key="seed.ungrd",
            transport=self.transport,
            details={
                "dataset": "ungrd_records",
                "municipality_count": len(self.seed["ungrdRecords"]),
            },
        )


ADAPTERS = {
    "IDEAM": IdeamSeedAdapter,
    "SGC": SgcSeedAdapter,
    "UNGRD": UngrdSeedAdapter,
}
