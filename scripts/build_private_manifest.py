import argparse
import json
from pathlib import Path

import pandas as pd

from seclog.data import sha256_file
from seclog.schemas import validate_test_frame, validate_training_frame


def build_manifest(train_path: Path, test_path: Path) -> dict[str, object]:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    validate_training_frame(train)
    validate_test_frame(test)
    return {
        "schema_version": 1,
        "train_sha256": sha256_file(train_path),
        "test_sha256": sha256_file(test_path),
        "train_rows": len(train),
        "test_rows": len(test),
        "has_anomaly_distribution": {
            str(key): int(value)
            for key, value in train["has_anomaly"].value_counts(dropna=False).sort_index().items()
        },
        "anomaly_type_distribution": {
            str(key): int(value)
            for key, value in train["primary_anomaly_type"].value_counts(dropna=False).items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a private data manifest without log text")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.train, args.test)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"manifest written: {args.output}")


if __name__ == "__main__":
    main()
