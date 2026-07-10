import argparse
import hashlib
import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from seclog.config import ProjectConfig, load_config
from seclog.constants import N_LABELS, N_TYPES, TYPE_TO_ID
from seclog.data import Sample, load_dataset, sha256_file
from seclog.decode import decode_logits_to_frame, samples_to_truth_frame
from seclog.features import nonempty_log_lines
from seclog.metrics import evaluate_predictions
from seclog.schemas import validate_training_frame
from seclog.splitting import add_template_groups, build_split_audit, make_locked_split
from seclog.training import Prediction, average_logits, run_training
from seclog.tuning import tune_decoder


def validate_legacy_run_args(run_args: dict[str, Any], config: ProjectConfig) -> None:
    expected = {
        "vocab_size": config.model.vocab_size,
        "emb_dim": config.model.emb_dim,
        "hidden": config.model.hidden,
        "layers": config.model.layers,
        "dropout": config.model.dropout,
        "max_tokens": config.features.max_tokens,
        "seed": config.training.seed,
        "folds": config.training.folds,
        "epochs": config.training.epochs,
        "batch_size": config.training.batch_size,
        "eval_batch_size": config.training.eval_batch_size,
        "lr": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
        "o_weight": config.training.o_weight,
        "boundary_loss_weight": config.training.boundary_loss_weight,
        "global_loss_weight": config.training.global_loss_weight,
        "patience": config.training.patience,
    }
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        actual = run_args.get(key)
        matches = (
            bool(np.isclose(actual, expected_value))
            if isinstance(expected_value, float) and actual is not None
            else actual == expected_value
        )
        if not matches:
            mismatches.append(key)
    legacy_seeds = [
        int(item.strip()) for item in str(run_args.get("seeds", "")).split(",") if item.strip()
    ]
    if legacy_seeds != config.training.seeds:
        mismatches.append("seeds")
    if mismatches:
        raise ValueError(f"legacy run arguments do not match config: {sorted(set(mismatches))}")


def normalize_oof_predictions(raw: object) -> dict[int, Prediction]:
    if not isinstance(raw, dict):
        raise ValueError("OOF artifact must be a mapping keyed by sample id")
    normalized: dict[int, Prediction] = {}
    for raw_id, raw_prediction in raw.items():
        if isinstance(raw_prediction, list):
            if not raw_prediction:
                raise ValueError(f"OOF artifact has no prediction for id {raw_id}")
            prediction = average_logits(raw_prediction)
        elif isinstance(raw_prediction, dict):
            prediction = raw_prediction
        else:
            raise ValueError(f"OOF prediction for id {raw_id} has an unsupported type")
        if set(prediction) != {"tag", "start", "end", "global"}:
            raise ValueError(f"OOF prediction heads are invalid for id {raw_id}")
        converted = {
            key: np.asarray(value, dtype=np.float32) for key, value in prediction.items()
        }
        if converted["tag"].ndim != 2 or converted["tag"].shape[1] != N_LABELS:
            raise ValueError(f"OOF tag shape is invalid for id {raw_id}")
        for head in ("start", "end"):
            if converted[head].shape != (converted["tag"].shape[0], N_TYPES):
                raise ValueError(f"OOF {head} shape is invalid for id {raw_id}")
        if converted["global"].shape != (N_TYPES + 1,):
            raise ValueError(f"OOF global shape is invalid for id {raw_id}")
        if not all(np.isfinite(value).all() for value in converted.values()):
            raise ValueError(f"OOF prediction contains non-finite values for id {raw_id}")
        normalized[int(raw_id)] = converted
    return normalized


def _samples_from_frame(frame: pd.DataFrame) -> list[Sample]:
    samples: list[Sample] = []
    for _, row in frame.iterrows():
        lines = nonempty_log_lines(row["log_text"])
        has_anomaly = int(row["has_anomaly"])
        primary_type = str(row["primary_anomaly_type"])
        primary = (
            (
                int(row["primary_start_idx"]),
                int(row["primary_end_idx"]),
                primary_type,
            )
            if has_anomaly
            else None
        )
        samples.append(
            Sample(
                sid=int(row["id"]),
                lines=lines,
                token_ids=[],
                has_anomaly=has_anomaly,
                primary=primary,
                global_label=TYPE_TO_ID[primary_type] if has_anomaly else N_TYPES,
            )
        )
    return samples


def _validate_oof_coverage(samples: list[Sample], predictions: dict[int, Prediction]) -> None:
    expected_ids = {sample.sid for sample in samples}
    if set(predictions) != expected_ids:
        missing = len(expected_ids - set(predictions))
        extra = len(set(predictions) - expected_ids)
        raise ValueError(f"OOF id coverage mismatch; missing={missing}, extra={extra}")
    for sample in samples:
        if predictions[sample.sid]["tag"].shape[0] != len(sample.lines):
            raise ValueError(f"OOF line count does not match id {sample.sid}")


def prediction_distribution(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        "has_anomaly": {
            str(key): int(value)
            for key, value in frame["has_anomaly"].value_counts().sort_index().items()
        },
        "primary_anomaly_type": {
            str(key): int(value)
            for key, value in frame["primary_anomaly_type"].value_counts().items()
        },
    }


def run_reproduction(
    data_dir: Path,
    config_path: Path,
    output_dir: Path,
    device_name: str,
    locked_seed: int,
    oof_path: Path | None = None,
    run_args_path: Path | None = None,
) -> tuple[Path, Path]:
    config = load_config(config_path)
    train_path = data_dir / "train.csv"
    frame = pd.read_csv(train_path)
    validate_training_frame(frame)
    grouped = add_template_groups(frame)
    tuning_frame, locked_frame = make_locked_split(
        grouped,
        test_size=0.20,
        random_state=locked_seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = build_split_audit(grouped, tuning_frame, locked_frame)
    audit.to_csv(output_dir / "split-audit.csv", index=False, encoding="utf-8")

    if (oof_path is None) != (run_args_path is None):
        raise ValueError("oof_path and run_args_path must be provided together")
    if oof_path is not None and run_args_path is not None:
        run_args = json.loads(run_args_path.read_text(encoding="utf-8"))
        validate_legacy_run_args(run_args, config)
        with oof_path.open("rb") as handle:
            oof_predictions = normalize_oof_predictions(pickle.load(handle))
        samples = _samples_from_frame(frame)
        _validate_oof_coverage(samples, oof_predictions)
        test_path = data_dir / "test.csv"
        dataset_payload = {
            "schema_version": 1,
            "train_sha256": sha256_file(train_path),
            "test_sha256": sha256_file(test_path),
            "train_rows": len(frame),
            "test_rows": len(pd.read_csv(test_path, usecols=["id"])),
            "vocab_size": config.model.vocab_size,
            "max_tokens": config.features.max_tokens,
        }
        checkpoint_count = int(run_args["folds"])
        prediction_source = {
            "kind": "verified_legacy_oof",
            "oof_sha256": sha256_file(oof_path),
            "run_args_sha256": sha256_file(run_args_path),
        }
    else:
        training = run_training(data_dir, output_dir / "training", config, device_name)
        # This pickle is loaded only from run_training in the same invocation.
        with training.oof_path.open("rb") as handle:
            oof_predictions = normalize_oof_predictions(pickle.load(handle))
        _train, _test, samples, _test_samples, dataset_manifest = load_dataset(
            data_dir,
            output_dir / "training" / "cache",
            config.model.vocab_size,
            config.features.max_tokens,
        )
        dataset_payload = asdict(dataset_manifest)
        checkpoint_count = len(training.checkpoints)
        prediction_source = {"kind": "fresh_training"}
    by_id = {sample.sid: sample for sample in samples}
    tuning_samples = [by_id[int(sample_id)] for sample_id in tuning_frame["id"]]
    locked_samples = [by_id[int(sample_id)] for sample_id in locked_frame["id"]]
    tuning_predictions = [oof_predictions[sample.sid] for sample in tuning_samples]
    locked_predictions = [oof_predictions[sample.sid] for sample in locked_samples]
    tuned_params = tune_decoder(
        tuning_samples,
        tuning_predictions,
        str(output_dir / "decoder-tuning"),
        length_stats={},
        fast=True,
        tune_mode="fast",
    )
    locked_output = decode_logits_to_frame(
        locked_samples,
        locked_predictions,
        tuned_params,
        length_stats={},
    )
    locked_metrics = evaluate_predictions(
        locked_output,
        samples_to_truth_frame(locked_samples),
    )
    metrics_path = output_dir / "reproduction-metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "locked_decoder_evaluation": locked_metrics,
                "decoder_tuning_rows": len(tuning_samples),
                "locked_evaluation_rows": len(locked_samples),
                "prediction_distribution": prediction_distribution(locked_output),
                "warning": (
                    "Locked only against decoder tuning; fold OOF models may use other locked rows."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    group_sizes = grouped.groupby("template_group").size()
    manifest_path = output_dir / "reproduction-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": dataset_payload,
                "config": asdict(config),
                "locked_seed": locked_seed,
                "template_groups": int(grouped["template_group"].nunique()),
                "duplicate_groups": int(group_sizes.gt(1).sum()),
                "largest_group": int(group_sizes.max()),
                "decoder_tuning_ids_sha256": _id_hash(tuning_frame["id"].tolist()),
                "locked_ids_sha256": _id_hash(locked_frame["id"].tolist()),
                "checkpoint_count": checkpoint_count,
                "prediction_source": prediction_source,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return metrics_path, manifest_path


def _id_hash(ids: list[object]) -> str:
    payload = ",".join(str(item) for item in sorted(ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a private locked decoder reproduction")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/final.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--locked-seed", type=int, default=20260711)
    parser.add_argument("--oof-path", type=Path)
    parser.add_argument("--run-args", type=Path)
    args = parser.parse_args()
    metrics_path, manifest_path = run_reproduction(
        args.data_dir,
        args.config,
        args.output_dir,
        args.device,
        args.locked_seed,
        args.oof_path,
        args.run_args,
    )
    print(f"metrics written: {metrics_path}")
    print(f"manifest written: {manifest_path}")


if __name__ == "__main__":
    main()
