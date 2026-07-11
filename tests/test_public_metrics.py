import pytest

from seclog.public_metrics import (
    apply_temperature,
    choose_f1_threshold,
    evaluate_sequence_predictions,
    evaluate_span_predictions,
    evaluate_normal_only_predictions,
    fit_temperature,
    inclusive_iou,
)
from seclog.public_protocol import (
    PreparedSample,
    PublicPrediction,
    PublicProtocolError,
    PublicSpan,
    TaskProfile,
    template_signature,
)


def _sequence_samples() -> tuple[PreparedSample, ...]:
    return tuple(
        PreparedSample(
            sid=str(index),
            lines=(f"line {index}",),
            has_anomaly=int(index % 2 == 1),
            spans=(),
            source_group=str(index),
            source_line_ids=(str(index),),
            template_key=template_signature((f"line {index}",)),
        )
        for index in range(4)
    )


def test_sequence_metrics_and_validation_threshold() -> None:
    samples = _sequence_samples()
    predictions = tuple(
        PublicPrediction(sample.sid, score, decision)
        for sample, score, decision in zip(samples, (0.1, 0.9, 0.2, 0.8), (0, 1, 0, 1))
    )
    metrics = evaluate_sequence_predictions(samples, predictions)
    assert metrics["f1"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert choose_f1_threshold((0.1, 0.9, 0.2, 0.8), (0, 1, 0, 1)) == 0.8


def test_span_metrics_match_inclusive_spans() -> None:
    sample = PreparedSample(
        sid="span",
        lines=("a", "b", "c", "d"),
        has_anomaly=1,
        spans=(PublicSpan(1, 2),),
        source_group="group",
        source_line_ids=("0", "1", "2", "3"),
        template_key="template",
    )
    predicted = PublicPrediction("span", 0.9, 1, (PublicSpan(1, 2),))
    metrics = evaluate_span_predictions((sample,), (predicted,))
    assert metrics["line_f1"] == 1.0
    assert metrics["span_f1"] == 1.0
    assert inclusive_iou(PublicSpan(1, 2), PublicSpan(2, 3)) == 1 / 3


def test_calibration_requires_validation_class_support() -> None:
    temperature = fit_temperature((0.2, 0.8, 0.3, 0.7), (0, 1, 0, 1))
    calibrated = apply_temperature((0.2, 0.8), temperature)
    assert len(calibrated) == 2
    with pytest.raises(PublicProtocolError, match="both binary classes"):
        choose_f1_threshold((0.2, 0.3), (0, 0))


def test_calibration_remains_finite_for_extreme_probabilities() -> None:
    temperature = fit_temperature((1e-6, 1 - 1e-6, 1e-5, 1 - 1e-5), (0, 1, 0, 1))
    calibrated = apply_temperature((1e-6, 1 - 1e-6), temperature)
    assert (calibrated > 0).all()
    assert (calibrated < 1).all()


def test_normal_only_metrics_report_false_positive_rates_only() -> None:
    samples = (
        PreparedSample(
            sid="normal",
            lines=("a", "b"),
            has_anomaly=0,
            spans=(),
            source_group="normal",
            source_line_ids=("0", "1"),
            template_key="normal",
        ),
    )
    predictions = (PublicPrediction("normal", 0.8, 1, (PublicSpan(1, 1),)),)
    metrics = evaluate_normal_only_predictions(samples, predictions, TaskProfile.SPAN_BINARY)
    assert metrics["sample_false_positive_rate"] == 1.0
    assert metrics["line_false_positive_rate"] == 0.5
