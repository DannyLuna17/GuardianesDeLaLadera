from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.datasets import export_seed_training_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the current bootstrap spatial training dataset as a versioned artifact."
    )
    parser.add_argument(
        "--version",
        default="training-spatial-seed-v1",
        help="Dataset version to export.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    dataset_path, dataset = export_seed_training_dataset(version=args.version)
    print(
        json.dumps(
            {
                "datasetPath": str(dataset_path),
                "datasetVersion": dataset["version"],
                "rows": dataset["summary"]["rows"],
                "featureCount": len(dataset["feature_order"]),
                "splitCounts": dataset["summary"]["splits"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
