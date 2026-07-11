import pytest

from seclog.public_protocol import PreparedSample, PublicProtocolError, PublicSpan
from seclog.public_splitting import (
    chronological_split,
    random_split,
    read_split_assignment,
    template_isolated_split,
    write_split_assignment,
)


def _samples() -> tuple[PreparedSample, ...]:
    items = []
    for index in range(18):
        anomaly = int(index % 2 == 0)
        lines = (f"2026-01-{index + 1:02d} service {index}",)
        items.append(
            PreparedSample(
                sid=str(index),
                lines=lines,
                has_anomaly=anomaly,
                spans=(PublicSpan(0, 0),) if anomaly else (),
                source_group=f"group-{index}",
                source_line_ids=(f"source-{index}",),
                template_key=f"template-{index}",
                timestamp=f"2026-01-{index + 1:02d}T00:00:00",
            )
        )
    return tuple(items)


def test_random_split_is_deterministic_and_line_disjoint(tmp_path) -> None:
    samples = _samples()
    assignment = random_split(samples, seed=7)
    assert assignment == random_split(samples, seed=7)
    assignment.validate(samples)
    path = tmp_path / "split.json"
    write_split_assignment(path, assignment)
    assert read_split_assignment(path, samples) == assignment


def test_chronological_split_keeps_later_examples_out_of_train() -> None:
    samples = _samples()
    assignment = chronological_split(samples)
    assert max(int(item) for item in assignment.train_ids) < min(
        int(item) for item in assignment.validation_ids
    )
    assert max(int(item) for item in assignment.validation_ids) < min(
        int(item) for item in assignment.test_ids
    )


def test_template_split_isolated_and_rejects_too_few_groups() -> None:
    samples = _samples()
    assignment = template_isolated_split(samples, seed=3)
    assignment.validate(samples)
    bad = tuple(
        PreparedSample(
            **{**sample.__dict__, "template_key": "same-template"}
        )
        for sample in samples
    )
    with pytest.raises(PublicProtocolError, match="template groups"):
        template_isolated_split(bad, seed=3)
