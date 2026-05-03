from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SyncResult:
    source_id: str
    processed_records: int
    status: str
    message: str
    adapter_key: str
    transport: str
    details: dict = field(default_factory=dict)
