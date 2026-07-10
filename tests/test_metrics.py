import pandas as pd
import pytest

from seclog.metrics import evaluate_predictions, span_iou


def test_span_iou_is_inclusive() -> None:
    assert span_iou(1, 3, 2, 4) == pytest.approx(2 / 4)
    assert span_iou(-1, -1, 0, 1) == 0


def test_perfect_predictions_score_one() -> None:
    from seclog.constants import ANOMALY_TYPES

    types = ["none", *ANOMALY_TYPES]
    frame = pd.DataFrame(
        {
            "id": list(range(len(types))),
            "has_anomaly": [0, *([1] * len(ANOMALY_TYPES))],
            "primary_start_idx": [-1, *([2] * len(ANOMALY_TYPES))],
            "primary_end_idx": [-1, *([4] * len(ANOMALY_TYPES))],
            "primary_anomaly_type": types,
            "all_spans": ["", *[f"2|4|{name}" for name in ANOMALY_TYPES]],
        }
    )
    result = evaluate_predictions(frame, frame)
    assert result == {
        "detect_f1": 1.0,
        "iou": 1.0,
        "type_f1": 1.0,
        "score": 1.0,
    }
