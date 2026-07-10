from pathlib import Path

from scripts.audit_publication import audit_paths


def test_audit_rejects_private_dataset_name(tmp_path: Path) -> None:
    bad = tmp_path / "train.csv"
    bad.write_text("id,log_text\n1,secret\n", encoding="utf-8")
    findings = audit_paths(tmp_path)
    assert any("train.csv" in finding for finding in findings)


def test_audit_ignores_coverage_cache(tmp_path: Path) -> None:
    (tmp_path / ".coverage").write_bytes(b"\x00\xffcoverage-cache")
    assert audit_paths(tmp_path) == []
