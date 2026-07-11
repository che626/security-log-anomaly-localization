import pytest

from seclog.public_protocol import PublicProtocolError
from seclog.public_reporting import PublicResultRecord, write_aggregate_report, write_result_record


def _record(model: str, manifest: str = "a" * 64) -> PublicResultRecord:
    return PublicResultRecord(
        experiment_id=f"hdfs-random-{model}",
        dataset="hdfs",
        profile="sequence_binary",
        split_strategy="random",
        model=model,
        manifest_sha256=manifest,
        metrics={"f1": 0.8, "precision": 0.9},
        metadata={"seed": 7},
    )


def test_aggregate_report_writes_table_summary_and_svg(tmp_path) -> None:
    result = write_aggregate_report(tmp_path, (_record("logistic"), _record("neural")))
    assert result["table"].is_file()
    assert "neural" in result["table"].read_text(encoding="utf-8")
    assert result["figure"].read_text(encoding="utf-8").startswith("<svg")
    write_result_record(tmp_path / "one.json", _record("logistic"))


def test_aggregate_report_rejects_mixed_manifests(tmp_path) -> None:
    with pytest.raises(PublicProtocolError, match="mixed manifest"):
        write_aggregate_report(tmp_path, (_record("one"), _record("two", "b" * 64)))
