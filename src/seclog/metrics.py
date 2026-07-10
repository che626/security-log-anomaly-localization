import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .constants import ANOMALY_TYPES


def span_iou(gold_start: int, gold_end: int, pred_start: int, pred_end: int) -> float:
    if min(gold_start, gold_end, pred_start, pred_end) < 0:
        return 0.0
    intersection = max(0, min(gold_end, pred_end) - max(gold_start, pred_start) + 1)
    union = max(gold_end, pred_end) - min(gold_start, pred_start) + 1
    return float(intersection / union) if union else 0.0


def evaluate_predictions(pred: pd.DataFrame, gold: pd.DataFrame) -> dict[str, float]:
    if pred["id"].tolist() != gold["id"].tolist():
        raise ValueError("prediction and gold ids must match in the same order")
    detect_f1 = f1_score(
        gold["has_anomaly"],
        pred["has_anomaly"],
        average="macro",
        labels=[0, 1],
        zero_division=0,
    )
    both = gold["has_anomaly"].eq(1) & pred["has_anomaly"].eq(1)
    ious = [
        span_iou(
            int(gold.loc[index, "primary_start_idx"]),
            int(gold.loc[index, "primary_end_idx"]),
            int(pred.loc[index, "primary_start_idx"]),
            int(pred.loc[index, "primary_end_idx"]),
        )
        for index in gold.index[both]
    ]
    iou = float(np.mean(ious)) if ious else 0.0
    type_f1 = (
        f1_score(
            gold.loc[both, "primary_anomaly_type"],
            pred.loc[both, "primary_anomaly_type"],
            average="macro",
            labels=ANOMALY_TYPES,
            zero_division=0,
        )
        if both.any()
        else 0.0
    )
    score = 0.15 * detect_f1 + 0.50 * iou + 0.35 * type_f1
    return {
        "detect_f1": float(detect_f1),
        "iou": iou,
        "type_f1": float(type_f1),
        "score": float(score),
    }
