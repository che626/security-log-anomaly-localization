import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from seclog.config import load_config
from seclog.data import load_dataset
from seclog.decode import decode_logits_to_frame, samples_to_truth_frame
from seclog.metrics import evaluate_predictions
from seclog.schemas import validate_training_frame
from seclog.splitting import add_template_groups, build_split_audit, make_locked_split
from seclog.training import Prediction, run_training
from seclog.tuning import tune_decoder


def run_reproduction(
    data_dir: Path,
    config_path: Path,
    output_dir: Path,
    device_name: str,
    locked_seed: int,
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

    training = run_training(data_dir, output_dir / "training", config, device_name)
    # This pickle is loaded only from run_training in the same invocation.
    with training.oof_path.open("rb") as handle:
        oof_predictions: dict[int, Prediction] = pickle.load(handle)
    _train, _test, samples, _test_samples, dataset_manifest = load_dataset(
        data_dir,
        output_dir / "training" / "cache",
        config.model.vocab_size,
        config.features.max_tokens,
    )
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
                "dataset": asdict(dataset_manifest),
                "config": asdict(config),
                "locked_seed": locked_seed,
                "template_groups": int(grouped["template_group"].nunique()),
                "duplicate_groups": int(group_sizes.gt(1).sum()),
                "largest_group": int(group_sizes.max()),
                "decoder_tuning_ids_sha256": _id_hash(tuning_frame["id"].tolist()),
                "locked_ids_sha256": _id_hash(locked_frame["id"].tolist()),
                "checkpoint_count": len(training.checkpoints),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return metrics_path, manifest_path


def _id_hash(ids: list[object]) -> str:
    import hashlib

    payload = ",".join(str(item) for item in sorted(ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a private locked decoder reproduction")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/final.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--locked-seed", type=int, default=20260711)
    args = parser.parse_args()
    metrics_path, manifest_path = run_reproduction(
        args.data_dir,
        args.config,
        args.output_dir,
        args.device,
        args.locked_seed,
    )
    print(f"metrics written: {metrics_path}")
    print(f"manifest written: {manifest_path}")


if __name__ == "__main__":
    main()
