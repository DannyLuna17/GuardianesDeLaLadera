from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings
from app.db.bootstrap import init_database
from app.db.session import session_scope
from app.services.structural_catalog import (
    ensure_real_data_structural_catalog,
    import_structural_catalog_bundle,
    load_structural_catalog_bundle,
)


def resolve_bundle_path(argv: list[str]) -> Path:
    settings = get_settings()
    if len(argv) > 2:
        raise SystemExit(
            "Usage: uv run python scripts/import_official_structural_catalog.py [bundle.json]"
        )
    if len(argv) == 2:
        return Path(argv[1]).resolve()
    return settings.resolved_structural_catalog_bundle_path


def main() -> None:
    bundle_path = resolve_bundle_path(sys.argv)
    bundle = load_structural_catalog_bundle(bundle_path)

    init_database()
    with session_scope() as session:
        counts = import_structural_catalog_bundle(session, bundle)
        ensure_real_data_structural_catalog(session, for_api=False)

    print(
        "Imported official structural catalog:",
        f"from {bundle_path}",
        f"{counts['municipalities']} municipalities,",
        f"{counts['zones']} zones,",
        f"{counts['road_segments']} road segments.",
    )


if __name__ == "__main__":
    main()
