import json
from functools import lru_cache
from typing import Any

from app.core.config import get_settings


@lru_cache
def load_seed_payload() -> dict[str, Any]:
    settings = get_settings()
    with settings.seed_data_path.open("r", encoding="utf-8") as file:
        return json.load(file)

