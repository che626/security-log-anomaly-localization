import json
from pathlib import Path

from seclog.features import clean_log_line, encode_log_line, hashed_bucket


def test_clean_log_line_matches_reference_cases() -> None:
    cases = json.loads(Path("tests/fixtures/feature_cases.json").read_text(encoding="utf-8"))
    for case in cases:
        assert clean_log_line(case["raw"]) == case["clean"]


def test_crc32_bucket_is_stable() -> None:
    assert hashed_bucket("w:host", 1024) == 61
    assert hashed_bucket("w:<ip>", 1024) == 401


def test_encode_empty_line_uses_nonzero_bucket() -> None:
    assert encode_log_line("", vocab_size=1024, max_tokens=8) == [756]


def test_encode_respects_max_tokens() -> None:
    assert len(encode_log_line("alpha beta gamma delta", 1024, max_tokens=3)) == 3
