import subprocess
import sys


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "seclog.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_help_lists_stable_commands() -> None:
    result = run_cli("--help")
    assert result.returncode == 0
    assert "check-data" in result.stdout
    assert "train" in result.stdout
    assert "predict" in result.stdout
    assert "evaluate" in result.stdout


def test_check_data_accepts_synthetic_fixture() -> None:
    result = run_cli(
        "check-data",
        "--train",
        "tests/fixtures/synthetic_data/train.csv",
        "--test",
        "tests/fixtures/synthetic_data/test.csv",
    )
    assert result.returncode == 0
    assert "schema: OK" in result.stdout
