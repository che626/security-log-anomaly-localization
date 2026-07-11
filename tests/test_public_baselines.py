from seclog.public_baselines import BASELINE_NAMES, run_baseline
from seclog.public_metrics import evaluate_sequence_predictions, evaluate_span_predictions
from seclog.public_protocol import PreparedSample, PublicSpan, TaskProfile, template_signature


def _sequence_samples(prefix: str, start: int, count: int) -> tuple[PreparedSample, ...]:
    samples = []
    for offset in range(count):
        anomaly = int((start + offset) % 2 == 1)
        marker = "failure critical" if anomaly else "service healthy"
        lines = (f"{marker} node {start + offset}",)
        samples.append(
            PreparedSample(
                sid=f"{prefix}-{offset}",
                lines=lines,
                has_anomaly=anomaly,
                spans=(),
                source_group=f"{prefix}-g-{offset}",
                source_line_ids=(f"{prefix}-source-{offset}",),
                template_key=template_signature(lines),
            )
        )
    return tuple(samples)


def _span_samples(prefix: str, start: int, count: int) -> tuple[PreparedSample, ...]:
    samples = []
    for offset in range(count):
        anomaly = int((start + offset) % 2 == 1)
        lines = ("service healthy", "failure critical" if anomaly else "service healthy")
        samples.append(
            PreparedSample(
                sid=f"{prefix}-{offset}",
                lines=lines,
                has_anomaly=anomaly,
                spans=(PublicSpan(1, 1),) if anomaly else (),
                source_group=f"{prefix}-g-{offset}",
                source_line_ids=(f"{prefix}-{offset}-0", f"{prefix}-{offset}-1"),
                template_key=template_signature(lines),
            )
        )
    return tuple(samples)


def test_all_sequence_baselines_fit_with_validation_only_thresholds() -> None:
    train = _sequence_samples("train", 0, 10)
    validation = _sequence_samples("validation", 10, 6)
    test = _sequence_samples("test", 20, 6)
    for name in BASELINE_NAMES:
        run = run_baseline(name, TaskProfile.SEQUENCE_BINARY, train, validation, test, seed=3)
        assert run.profile == "sequence_binary"
        assert len(run.test_predictions) == len(test)
        assert evaluate_sequence_predictions(test, run.test_predictions)["sample_count"] == len(test)


def test_all_span_baselines_return_binary_spans() -> None:
    train = _span_samples("train", 0, 10)
    validation = _span_samples("validation", 10, 6)
    test = _span_samples("test", 20, 6)
    for name in BASELINE_NAMES:
        run = run_baseline(name, TaskProfile.SPAN_BINARY, train, validation, test, seed=4)
        assert len(run.test_predictions) == len(test)
        assert evaluate_span_predictions(test, run.test_predictions)["sample_count"] == len(test)
