from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.training import export_seed_linear_artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and export the seed-backed spatial model artifact.")
    parser.add_argument(
        "--version",
        default="trained-spatial-seed-v1",
        help="Artifact version to export.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.75,
        help="Ridge regularization strength.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    artifact_path, artifact = export_seed_linear_artifact(version=args.version, alpha=args.alpha)
    print(
        json.dumps(
            {
                "artifactPath": str(artifact_path),
                "modelVersion": artifact["version"],
                "rows": artifact["training"]["rows"],
                "metrics": artifact["training"]["metrics"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
