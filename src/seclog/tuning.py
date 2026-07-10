import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .data import Sample
from .decode import decode_logits_to_frame, samples_to_truth_frame
from .metrics import evaluate_predictions

DecoderParams = dict[str, Any]
Prediction = dict[str, np.ndarray]


def make_decoder_grid(mode: str = "smart") -> list[DecoderParams]:
    mode = str(mode).lower()
    if mode == "fast":
        axes = (
            [-0.15, 0.0, 0.12],
            [0.10, 0.20, 0.30],
            [0, 1],
            ["earliest_score", "score"],
            [1],
            [0.35],
            [0.10],
            [0.08],
        )
    elif mode == "smart":
        axes = (
            [-0.25, -0.15, 0.0, 0.12, 0.22],
            [0.05, 0.15, 0.25, 0.35],
            [0, 1],
            ["earliest_score", "score"],
            [1, 2],
            [0.25, 0.45],
            [0.10],
            [0.08],
        )
    elif mode == "v3":
        axes = (
            [-0.22, -0.12, 0.0, 0.10, 0.20],
            [0.05, 0.12, 0.18, 0.25, 0.32],
            [0, 1, 2],
            ["earliest_score", "score"],
            [1, 2],
            [0.25, 0.35, 0.45],
            [0.08, 0.12],
            [0.06, 0.10],
        )
    else:
        axes = (
            [-0.30, -0.15, 0.0, 0.12, 0.25],
            [0.00, 0.10, 0.20, 0.30, 0.40],
            [0, 1, 2],
            ["earliest", "earliest_score", "score"],
            [1, 2],
            [0.20, 0.35, 0.55],
            [0.10],
            [0.08],
        )
    grid: list[DecoderParams] = []
    for values in product(*axes):
        (
            anomaly_bias,
            min_conf,
            bridge_gap,
            strategy,
            refine_radius,
            boundary_weight,
            global_weight,
            length_weight,
        ) = values
        grid.append(
            {
                "anomaly_bias": anomaly_bias,
                "min_conf": min_conf,
                "bridge_gap": bridge_gap,
                "edge_prob": 0.10,
                "switch_penalty": -1.8,
                "continue_bonus": 0.20,
                "min_len": 1,
                "hard_max_len": 18,
                "long_span_conf": 0.58,
                "primary_strategy": strategy,
                "refine_radius": refine_radius,
                "boundary_weight": boundary_weight,
                "global_weight": global_weight,
                "length_weight": length_weight,
                "use_boundary_candidates": False,
            }
        )
    endpoint_confidence = [0.15, 0.25] if mode != "fast" else [0.20]
    endpoint_start = [0.45, 0.50] if mode == "v3" else [0.50]
    endpoint_support = [0.12, 0.16] if mode == "v3" else [0.16]
    for anomaly_bias, min_conf, start_threshold, tag_support in product(
        [-0.15, 0.0, 0.12], endpoint_confidence, endpoint_start, endpoint_support
    ):
        grid.append(
            {
                "anomaly_bias": anomaly_bias,
                "min_conf": min_conf,
                "bridge_gap": 1,
                "edge_prob": 0.10,
                "switch_penalty": -1.8,
                "continue_bonus": 0.20,
                "min_len": 1,
                "hard_max_len": 18,
                "long_span_conf": 0.60,
                "primary_strategy": "earliest_score",
                "refine_radius": 1,
                "boundary_weight": 0.45,
                "global_weight": 0.12,
                "length_weight": 0.08,
                "use_boundary_candidates": True,
                "boundary_start_th": start_threshold,
                "boundary_end_th": start_threshold,
                "candidate_tag_support": tag_support,
                "candidate_global_min": 0.05,
                "candidate_max_len": 12,
            }
        )
    return _unique_params(grid)


def _unique_params(candidates: list[DecoderParams]) -> list[DecoderParams]:
    seen: set[tuple[tuple[str, Any], ...]] = set()
    unique: list[DecoderParams] = []
    for params in candidates:
        key = tuple(sorted(params.items()))
        if key not in seen:
            seen.add(key)
            unique.append(params)
    return unique


def _bounded_local_params(base: DecoderParams, updates: DecoderParams) -> DecoderParams:
    item = dict(base)
    item.update(updates)
    item["min_conf"] = float(min(0.60, max(0.0, item.get("min_conf", 0.20))))
    item["anomaly_bias"] = float(min(0.45, max(-0.45, item.get("anomaly_bias", 0.0))))
    item["bridge_gap"] = int(min(3, max(0, item.get("bridge_gap", 1))))
    item["refine_radius"] = int(min(3, max(0, item.get("refine_radius", 1))))
    return item


def make_v3_local_grid(best_params: DecoderParams) -> list[DecoderParams]:
    base = dict(best_params)
    candidates: list[DecoderParams] = [_bounded_local_params(base, {})]

    def add_values(name: str, values: list[Any]) -> None:
        candidates.extend(_bounded_local_params(base, {name: value}) for value in values)

    add_values(
        "min_conf", [base.get("min_conf", 0.20) + delta for delta in [-0.06, -0.03, 0.03, 0.06]]
    )
    add_values(
        "anomaly_bias",
        [base.get("anomaly_bias", 0.0) + delta for delta in [-0.06, -0.03, 0.03, 0.06]],
    )
    add_values("bridge_gap", [0, 1, 2, 3])
    add_values("refine_radius", [0, 1, 2, 3])
    add_values("switch_penalty", [-2.3, -2.0, -1.8, -1.55, -1.3])
    add_values("continue_bonus", [0.10, 0.16, 0.20, 0.26, 0.32])
    add_values("boundary_weight", [0.20, 0.30, 0.40, 0.50, 0.60])
    add_values("global_weight", [0.04, 0.08, 0.12, 0.16])
    add_values("length_weight", [0.04, 0.08, 0.12])
    add_values("long_span_conf", [0.52, 0.58, 0.62, 0.68])
    add_values("primary_strategy", ["earliest", "earliest_score", "score", "longest"])
    add_values("use_boundary_candidates", [False, True])
    candidates.extend(
        [
            _bounded_local_params(
                base,
                {
                    "min_conf": base.get("min_conf", 0.20) + 0.04,
                    "anomaly_bias": base.get("anomaly_bias", 0.0) - 0.03,
                    "switch_penalty": -2.0,
                },
            ),
            _bounded_local_params(
                base,
                {
                    "min_conf": base.get("min_conf", 0.20) - 0.04,
                    "anomaly_bias": base.get("anomaly_bias", 0.0) + 0.03,
                    "bridge_gap": 2,
                },
            ),
            _bounded_local_params(
                base, {"boundary_weight": 0.55, "refine_radius": 2, "long_span_conf": 0.62}
            ),
            _bounded_local_params(
                base,
                {"boundary_weight": 0.30, "global_weight": 0.16, "primary_strategy": "score"},
            ),
        ]
    )
    return _unique_params(candidates)


def make_v4_local_grid(best_params: DecoderParams) -> list[DecoderParams]:
    base = dict(best_params)
    candidates = [_bounded_local_params(base, {"post_adjust_mode": "none"})]
    for short_length, confidence, neighbor in product(
        [1, 2, 3], [0.55, 0.75, 1.01], [0.00, 0.03, 0.06, 0.10]
    ):
        common = {
            "post_adjust_mode": "short_expand",
            "post_short_max_len": short_length,
            "post_short_max_conf": confidence,
            "post_neighbor_support": neighbor,
            "post_expand_right": 1,
        }
        candidates.append(_bounded_local_params(base, {**common, "post_expand_left": 0}))
        candidates.append(_bounded_local_params(base, {**common, "post_expand_left": 1}))
    for long_length, confidence in product([9, 10, 11, 12], [0.75, 1.01]):
        candidates.append(
            _bounded_local_params(
                base,
                {
                    "post_adjust_mode": "long_shrink",
                    "post_long_min_len": long_length,
                    "post_long_max_conf": confidence,
                    "post_shrink_left": 0,
                    "post_shrink_right": 1,
                },
            )
        )
    candidates.extend(
        [
            _bounded_local_params(
                base,
                {
                    "post_adjust_mode": "both",
                    "post_short_max_len": 2,
                    "post_short_max_conf": 1.01,
                    "post_neighbor_support": 0.03,
                    "post_expand_left": 0,
                    "post_expand_right": 1,
                    "post_long_min_len": 10,
                    "post_long_max_conf": 1.01,
                    "post_shrink_right": 1,
                    "boundary_weight": 0.40,
                    "length_weight": 0.10,
                },
            ),
            _bounded_local_params(
                base,
                {
                    "post_adjust_mode": "both",
                    "post_short_max_len": 2,
                    "post_short_max_conf": 0.75,
                    "post_neighbor_support": 0.06,
                    "post_expand_left": 1,
                    "post_expand_right": 1,
                    "post_long_min_len": 10,
                    "post_long_max_conf": 0.75,
                    "post_shrink_right": 1,
                    "boundary_weight": 0.45,
                    "length_weight": 0.08,
                },
            ),
        ]
    )
    return _unique_params(candidates)


def make_stratified_eval_slice(
    samples: list[Sample],
    pred_list: list[Prediction],
    max_items: int,
    seed: int = 20260504,
) -> tuple[list[Sample], list[Prediction]]:
    if max_items <= 0 or max_items >= len(samples):
        return samples, pred_list
    generator = np.random.default_rng(seed)
    groups: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        if sample.has_anomaly == 0:
            key = "none"
        else:
            if sample.primary is None:
                raise ValueError(f"anomalous sample {sample.sid} is missing its primary span")
            key = sample.primary[2]
        groups.setdefault(key, []).append(index)
    chosen: list[int] = []
    for indices in groups.values():
        count = max(1, int(round(max_items * len(indices) / len(samples))))
        count = min(count, len(indices))
        chosen.extend(generator.choice(indices, size=count, replace=False).tolist())
    if len(chosen) > max_items:
        chosen = generator.choice(chosen, size=max_items, replace=False).tolist()
    chosen = sorted(chosen)
    return [samples[index] for index in chosen], [pred_list[index] for index in chosen]


def _evaluate_grid(
    samples: list[Sample],
    predictions: list[Prediction],
    params_grid: list[DecoderParams],
    length_stats: dict[str, dict[str, float]],
    description: str,
) -> tuple[list[dict[str, Any]], tuple[dict[str, float], DecoderParams] | None]:
    gold = samples_to_truth_frame(samples)
    rows: list[dict[str, Any]] = []
    best: tuple[dict[str, float], DecoderParams] | None = None
    for params in tqdm(params_grid, desc=description):
        prediction = decode_logits_to_frame(samples, predictions, params, length_stats)
        score = evaluate_predictions(prediction, gold)
        rows.append({**params, **score})
        if best is None or score["score"] > best[0]["score"]:
            best = (score, params)
    return rows, best


def tune_decoder(
    samples: list[Sample],
    pred_list: list[Prediction],
    out_dir: str,
    length_stats: dict[str, dict[str, float]],
    fast: bool = False,
    tune_mode: str = "staged",
    tune_subset: int = 3000,
    top_k_full: int = 24,
) -> DecoderParams:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preset: DecoderParams = {
        "anomaly_bias": 0.0,
        "min_conf": 0.20,
        "bridge_gap": 1,
        "edge_prob": 0.10,
        "switch_penalty": -1.8,
        "continue_bonus": 0.20,
        "min_len": 1,
        "hard_max_len": 18,
        "long_span_conf": 0.58,
        "primary_strategy": "earliest_score",
        "refine_radius": 1,
        "boundary_weight": 0.35,
        "global_weight": 0.10,
        "length_weight": 0.08,
        "use_boundary_candidates": False,
    }
    if tune_mode == "none":
        (output_dir / "best_decode_params.json").write_text(
            json.dumps(
                {"params": preset, "score": {"note": "preset_no_tune"}},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return preset
    grid_mode = (
        "fast"
        if fast or tune_mode == "fast"
        else "full"
        if tune_mode == "full"
        else "v3"
        if tune_mode == "v3"
        else "smart"
    )
    grid = make_decoder_grid(grid_mode)
    if tune_mode in {"staged", "v3", "v4"} and len(samples) > tune_subset:
        subset_samples, subset_predictions = make_stratified_eval_slice(
            samples, pred_list, tune_subset
        )
        stage_rows, _ = _evaluate_grid(
            subset_samples, subset_predictions, grid, length_stats, "tune-stage1"
        )
        stage_frame = pd.DataFrame(stage_rows).sort_values("score", ascending=False)
        stage_frame.to_csv(output_dir / "decode_tuning_stage1_subset.csv", index=False)
        metric_columns = ["detect_f1", "iou", "type_f1", "score"]
        shortlist = (
            stage_frame.head(top_k_full)
            .drop(columns=[column for column in metric_columns if column in stage_frame.columns])
            .to_dict("records")
        )
        full_rows, best = _evaluate_grid(
            samples, pred_list, shortlist, length_stats, "tune-stage2"
        )
    else:
        full_rows, best = _evaluate_grid(samples, pred_list, grid, length_stats, "tune")
    if best is None:
        raise ValueError("decoder tuning grid is empty")
    result_frame = pd.DataFrame(full_rows).sort_values("score", ascending=False)
    if tune_mode in {"v3", "v4"}:
        local_grid = (
            make_v3_local_grid(best[1]) if tune_mode == "v3" else make_v4_local_grid(best[1])
        )
        local_rows, local_best = _evaluate_grid(
            samples, pred_list, local_grid, length_stats, f"tune-{tune_mode}-local"
        )
        local_frame = pd.DataFrame(local_rows).sort_values("score", ascending=False)
        local_frame.to_csv(output_dir / f"decode_tuning_{tune_mode}_local.csv", index=False)
        if local_best is not None and local_best[0]["score"] > best[0]["score"]:
            best = local_best
        result_frame = pd.concat([result_frame, local_frame], ignore_index=True).sort_values(
            "score", ascending=False
        )
    result_frame.to_csv(output_dir / "decode_tuning.csv", index=False)
    best_score, best_params = best
    (output_dir / "best_decode_params.json").write_text(
        json.dumps({"params": best_params, "score": best_score}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return best_params
