import argparse
import json
from pathlib import Path

import pandas as pd

from .config import load_config
from .data import sha256_file
from .inference import predict
from .metrics import evaluate_predictions
from .public_baselines import BASELINE_NAMES, run_baseline
from .public_data import (
    prepare_bgl,
    prepare_hdfs,
    prepare_openstack,
    prepare_openstack_raw,
    prepare_thunderbird,
)
from .public_metrics import (
    evaluate_normal_only_predictions,
    evaluate_sequence_predictions,
    evaluate_span_predictions,
)
from .public_protocol import TaskProfile, read_manifest, read_prepared_dataset, sha256_file as public_sha256
from .public_reporting import (
    PublicResultRecord,
    read_result_record,
    write_aggregate_report,
    write_predictions,
    write_result_record,
)
from .public_splitting import (
    chronological_split,
    partition_samples,
    random_split,
    read_split_assignment,
    template_isolated_split,
    write_split_assignment,
)
from .public_training import load_public_neural_config, predict_public_neural, train_public_neural
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


def _public_prepare(args: argparse.Namespace) -> None:
    if args.dataset == "hdfs":
        if args.logs is None or args.labels is None:
            raise ValueError("HDFS preparation requires --logs and --labels")
        paths = prepare_hdfs(args.logs, args.labels, args.output_dir)
    elif args.dataset == "openstack":
        if args.labels is None:
            raise ValueError("OpenStack preparation requires --labels")
        if args.abnormal_log is not None or args.normal_log:
            if args.abnormal_log is None or not args.normal_log:
                raise ValueError("OpenStack raw preparation requires --abnormal-log and --normal-log")
            paths = prepare_openstack_raw(args.normal_log, args.abnormal_log, args.labels, args.output_dir)
        else:
            if args.logs is None:
                raise ValueError("grouped-CSV OpenStack preparation requires --logs")
            columns = {
                "group_column": args.group_column,
                "message_column": args.message_column,
                "label_group_column": args.label_group_column,
                "label_column": args.label_column,
            }
            paths = prepare_openstack(args.logs, args.labels, args.output_dir, **columns)
    elif args.dataset == "bgl":
        if args.logs is None:
            raise ValueError("BGL preparation requires --logs")
        paths = prepare_bgl(
            args.logs,
            args.output_dir,
            window_size=args.window_size,
            stride=args.stride,
            max_source_lines=args.max_source_lines,
        )
    elif args.dataset == "thunderbird":
        if args.logs is None or args.max_source_lines is None:
            raise ValueError("Thunderbird preparation requires --logs and --max-source-lines")
        paths = prepare_thunderbird(
            args.logs,
            args.output_dir,
            window_size=args.window_size,
            stride=args.stride,
            max_source_lines=args.max_source_lines,
        )
    else:
        raise ValueError(f"unsupported public dataset {args.dataset}")
    print(
        json.dumps(
            {"prepared_dataset": str(paths.dataset_path), "manifest": str(paths.manifest_path)},
            ensure_ascii=False,
            indent=2,
        )
    )


def _public_split(args: argparse.Namespace) -> None:
    profile = TaskProfile(args.profile)
    samples = read_prepared_dataset(args.prepared, profile)
    kwargs = {"seed": args.seed, "train_fraction": args.train_fraction, "validation_fraction": args.validation_fraction}
    if args.strategy == "random":
        assignment = random_split(samples, **kwargs)
    elif args.strategy == "chronological":
        assignment = chronological_split(samples, **kwargs)
    elif args.strategy == "template_isolated":
        assignment = template_isolated_split(samples, **kwargs)
    else:
        raise ValueError(f"unsupported public split strategy {args.strategy}")
    write_split_assignment(args.output, assignment)
    print(json.dumps({"split": str(args.output), "strategy": assignment.strategy}, ensure_ascii=False, indent=2))


def _public_context(args: argparse.Namespace):
    profile = TaskProfile(args.profile)
    manifest = read_manifest(args.manifest)
    if manifest.profile != profile.value:
        raise ValueError("manifest profile does not match --profile")
    samples = read_prepared_dataset(args.prepared, profile)
    assignment = read_split_assignment(args.split, samples)
    return profile, manifest, public_sha256(args.manifest), assignment, partition_samples(samples, assignment)


def _public_run_baseline(args: argparse.Namespace) -> None:
    profile, manifest, manifest_hash, assignment, partitions = _public_context(args)
    train, validation, test = partitions
    run = run_baseline(args.name, profile, train, validation, test, seed=args.seed)
    metrics = (
        evaluate_sequence_predictions(test, run.test_predictions)
        if profile is TaskProfile.SEQUENCE_BINARY
        else evaluate_span_predictions(test, run.test_predictions)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiment_id = args.experiment_id or f"{manifest.dataset}-{assignment.strategy}-{run.name}"
    prediction_path = args.output_dir / f"{experiment_id}-predictions.jsonl"
    record_path = args.output_dir / f"{experiment_id}-result.json"
    write_predictions(prediction_path, run.test_predictions)
    write_result_record(
        record_path,
        PublicResultRecord(
            experiment_id=experiment_id,
            dataset=manifest.dataset,
            profile=profile.value,
            split_strategy=assignment.strategy,
            model=run.name,
            manifest_sha256=manifest_hash,
            metrics=metrics,
            metadata={**run.metadata, "threshold": run.threshold, "validation_sample_count": len(validation)},
        ),
    )
    print(json.dumps({"result": str(record_path), "predictions": str(prediction_path)}, ensure_ascii=False, indent=2))


def _public_train(args: argparse.Namespace) -> None:
    profile, manifest, manifest_hash, assignment, partitions = _public_context(args)
    train, validation, test = partitions
    config = load_public_neural_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiment_id = args.experiment_id or f"{manifest.dataset}-{assignment.strategy}-neural"
    checkpoint = args.output_dir / f"{experiment_id}.pt"
    run = train_public_neural(
        profile,
        train,
        validation,
        test,
        config,
        device_name=args.device,
        checkpoint_path=checkpoint,
        manifest_sha256=manifest_hash,
    )
    metrics = (
        evaluate_sequence_predictions(test, run.test_predictions)
        if profile is TaskProfile.SEQUENCE_BINARY
        else evaluate_span_predictions(test, run.test_predictions)
    )
    prediction_path = args.output_dir / f"{experiment_id}-predictions.jsonl"
    record_path = args.output_dir / f"{experiment_id}-result.json"
    write_predictions(prediction_path, run.test_predictions)
    write_result_record(
        record_path,
        PublicResultRecord(
            experiment_id=experiment_id,
            dataset=manifest.dataset,
            profile=profile.value,
            split_strategy=assignment.strategy,
            model="neural_cnn_bigru",
            manifest_sha256=manifest_hash,
            metrics=metrics,
            metadata={
                **run.metadata,
                "threshold": run.threshold,
                "temperature": run.temperature,
                "epochs_trained": run.epochs_trained,
                "validation_sample_count": len(validation),
            },
        ),
    )
    print(
        json.dumps(
            {"result": str(record_path), "predictions": str(prediction_path), "checkpoint": str(checkpoint)},
            ensure_ascii=False,
            indent=2,
        )
    )


def _public_report(args: argparse.Namespace) -> None:
    records = [read_result_record(path) for path in args.result]
    paths = write_aggregate_report(args.output_dir, records)
    print(json.dumps({name: str(path) for name, path in paths.items()}, ensure_ascii=False, indent=2))


def _public_transfer_negative(args: argparse.Namespace) -> None:
    profile = TaskProfile(args.profile)
    target_manifest = read_manifest(args.target_manifest)
    if target_manifest.profile != profile.value:
        raise ValueError("target manifest profile does not match --profile")
    target_samples = read_prepared_dataset(args.target_prepared, profile)
    source_manifest_hash = public_sha256(args.source_manifest)
    source_result = read_result_record(args.source_result)
    if source_result.profile != profile.value:
        raise ValueError("source result profile does not match --profile")
    try:
        threshold = float(source_result.metadata["threshold"])
        temperature = float(source_result.metadata["temperature"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("source result lacks neural threshold or temperature metadata") from exc
    predictions = predict_public_neural(
        profile,
        target_samples,
        load_public_neural_config(args.config),
        checkpoint_path=args.checkpoint,
        source_manifest_sha256=source_manifest_hash,
        threshold=threshold,
        temperature=temperature,
        device_name=args.device,
    )
    metrics = evaluate_normal_only_predictions(target_samples, predictions, profile)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiment_id = args.experiment_id or f"{source_result.dataset}-to-{target_manifest.dataset}-normal-only"
    prediction_path = args.output_dir / f"{experiment_id}-predictions.jsonl"
    result_path = args.output_dir / f"{experiment_id}-result.json"
    write_predictions(prediction_path, predictions)
    write_result_record(
        result_path,
        PublicResultRecord(
            experiment_id=experiment_id,
            dataset=target_manifest.dataset,
            profile=profile.value,
            split_strategy="cross_system_normal_only",
            model=f"{source_result.model}_from_{source_result.dataset}",
            manifest_sha256=public_sha256(args.target_manifest),
            metrics=metrics,
            metadata={
                "source_dataset": source_result.dataset,
                "source_experiment_id": source_result.experiment_id,
                "source_manifest_sha256": source_manifest_hash,
                "threshold_from_source_validation": threshold,
                "temperature_from_source_validation": temperature,
                "device": args.device,
            },
        ),
    )
    print(json.dumps({"result": str(result_path), "predictions": str(prediction_path)}, ensure_ascii=False, indent=2))


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

    public_prepare = commands.add_parser("public-prepare", help="prepare a local public Loghub dataset")
    public_prepare.add_argument("--dataset", choices=("hdfs", "openstack", "bgl", "thunderbird"), required=True)
    public_prepare.add_argument("--logs", type=Path)
    public_prepare.add_argument("--labels", type=Path)
    public_prepare.add_argument("--normal-log", type=Path, action="append")
    public_prepare.add_argument("--abnormal-log", type=Path)
    public_prepare.add_argument("--output-dir", type=Path, required=True)
    public_prepare.add_argument("--window-size", type=int, default=64)
    public_prepare.add_argument("--stride", type=int, default=64)
    public_prepare.add_argument("--max-source-lines", type=int)
    public_prepare.add_argument("--group-column")
    public_prepare.add_argument("--message-column")
    public_prepare.add_argument("--label-group-column")
    public_prepare.add_argument("--label-column")
    public_prepare.set_defaults(handler=_public_prepare)

    public_split = commands.add_parser("public-split", help="create a leakage-safe public split")
    public_split.add_argument("--prepared", type=Path, required=True)
    public_split.add_argument("--profile", choices=[item.value for item in TaskProfile], required=True)
    public_split.add_argument("--strategy", choices=("random", "chronological", "template_isolated"), required=True)
    public_split.add_argument("--output", type=Path, required=True)
    public_split.add_argument("--seed", type=int, default=20260711)
    public_split.add_argument("--train-fraction", type=float, default=0.6)
    public_split.add_argument("--validation-fraction", type=float, default=0.2)
    public_split.set_defaults(handler=_public_split)

    public_baseline = commands.add_parser("public-run-baseline", help="run a public baseline")
    public_baseline.add_argument("--prepared", type=Path, required=True)
    public_baseline.add_argument("--manifest", type=Path, required=True)
    public_baseline.add_argument("--split", type=Path, required=True)
    public_baseline.add_argument("--profile", choices=[item.value for item in TaskProfile], required=True)
    public_baseline.add_argument("--name", choices=BASELINE_NAMES, required=True)
    public_baseline.add_argument("--output-dir", type=Path, required=True)
    public_baseline.add_argument("--experiment-id")
    public_baseline.add_argument("--seed", type=int, default=20260711)
    public_baseline.set_defaults(handler=_public_run_baseline)

    public_train = commands.add_parser("public-train", help="train the binary public neural profile")
    public_train.add_argument("--prepared", type=Path, required=True)
    public_train.add_argument("--manifest", type=Path, required=True)
    public_train.add_argument("--split", type=Path, required=True)
    public_train.add_argument("--profile", choices=[item.value for item in TaskProfile], required=True)
    public_train.add_argument("--config", type=Path, required=True)
    public_train.add_argument("--output-dir", type=Path, required=True)
    public_train.add_argument("--experiment-id")
    public_train.add_argument("--device", default="auto")
    public_train.set_defaults(handler=_public_train)

    public_report = commands.add_parser("public-report", help="aggregate compatible public result records")
    public_report.add_argument("--result", type=Path, action="append", required=True)
    public_report.add_argument("--output-dir", type=Path, required=True)
    public_report.set_defaults(handler=_public_report)

    public_transfer = commands.add_parser(
        "public-transfer-negative",
        help="measure false positives from a source neural model on an all-normal target corpus",
    )
    public_transfer.add_argument("--target-prepared", type=Path, required=True)
    public_transfer.add_argument("--target-manifest", type=Path, required=True)
    public_transfer.add_argument("--source-manifest", type=Path, required=True)
    public_transfer.add_argument("--source-result", type=Path, required=True)
    public_transfer.add_argument("--checkpoint", type=Path, required=True)
    public_transfer.add_argument("--config", type=Path, required=True)
    public_transfer.add_argument("--profile", choices=[item.value for item in TaskProfile], required=True)
    public_transfer.add_argument("--output-dir", type=Path, required=True)
    public_transfer.add_argument("--experiment-id")
    public_transfer.add_argument("--device", default="auto")
    public_transfer.set_defaults(handler=_public_transfer_negative)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
