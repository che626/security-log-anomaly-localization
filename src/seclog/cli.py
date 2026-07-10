import argparse
import json
from pathlib import Path

import pandas as pd

from .config import load_config
from .data import sha256_file
from .inference import predict
from .metrics import evaluate_predictions
from .schemas import validate_prediction_frame, validate_test_frame, validate_training_frame
from .training import run_training


def _check_data(args: argparse.Namespace) -> None:
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    validate_training_frame(train)
    validate_test_frame(test)
    summary = {
        "train_sha256": sha256_file(args.train),
        "test_sha256": sha256_file(args.test),
        "train_rows": len(train),
        "test_rows": len(test),
        "has_anomaly_distribution": {
            str(key): int(value)
            for key, value in train["has_anomaly"].value_counts().sort_index().items()
        },
        "anomaly_type_distribution": {
            str(key): int(value)
            for key, value in train["primary_anomaly_type"].value_counts().items()
        },
    }
    print("schema: OK")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _train(args: argparse.Namespace) -> None:
    result = run_training(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config=load_config(args.config),
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "checkpoints": [str(item.path) for item in result.checkpoints],
                "oof_path": str(result.oof_path),
                "metrics_path": str(result.metrics_path),
                "manifest_path": str(result.manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _predict(args: argparse.Namespace) -> None:
    output = predict(
        test_path=args.test,
        checkpoint_paths=args.checkpoint,
        config=load_config(args.config),
        output_path=args.output,
        device_name=args.device,
        allow_degenerate=args.allow_degenerate,
    )
    print(f"prediction rows: {len(output)}")
    print(f"submission written: {args.output}")


def _evaluate(args: argparse.Namespace) -> None:
    prediction = pd.read_csv(args.prediction)
    gold = pd.read_csv(args.gold)
    validate_training_frame(gold)
    validate_prediction_frame(prediction, expected_ids=gold["id"].tolist())
    print(json.dumps(evaluate_predictions(prediction, gold), ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seclog",
        description="Security-log anomaly localization training and inference",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    check_data = commands.add_parser("check-data", help="validate explicit train/test CSVs")
    check_data.add_argument("--train", type=Path, required=True)
    check_data.add_argument("--test", type=Path, required=True)
    check_data.set_defaults(handler=_check_data)

    train = commands.add_parser("train", help="train folds from an explicit data directory")
    train.add_argument("--data-dir", type=Path, required=True)
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--device", default="auto")
    train.set_defaults(handler=_train)

    predict_parser = commands.add_parser("predict", help="run validated fold inference")
    predict_parser.add_argument("--test", type=Path, required=True)
    predict_parser.add_argument("--config", type=Path, required=True)
    predict_parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.add_argument("--device", default="auto")
    predict_parser.add_argument("--allow-degenerate", action="store_true")
    predict_parser.set_defaults(handler=_predict)

    evaluate = commands.add_parser("evaluate", help="score a prediction CSV")
    evaluate.add_argument("--prediction", type=Path, required=True)
    evaluate.add_argument("--gold", type=Path, required=True)
    evaluate.set_defaults(handler=_evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
