import subprocess
import sys


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "seclog.cli", *args], capture_output=True, text=True, check=False
    )


def test_public_cli_runs_preparation_baseline_neural_and_report(tmp_path) -> None:
    prepared_dir = tmp_path / "prepared"
    prepared = prepared_dir / "bgl-span_binary.jsonl"
    manifest = prepared_dir / "bgl-span_binary.manifest.json"
    split = tmp_path / "split.json"
    output = tmp_path / "outputs"
    fixture = "tests/fixtures/public_logs/synthetic_bgl.log"
    result = _run(
        "public-prepare",
        "--dataset",
        "bgl",
        "--logs",
        fixture,
        "--output-dir",
        str(prepared_dir),
        "--window-size",
        "2",
        "--stride",
        "2",
    )
    assert result.returncode == 0, result.stderr
    result = _run(
        "public-split",
        "--prepared",
        str(prepared),
        "--profile",
        "span_binary",
        "--strategy",
        "random",
        "--output",
        str(split),
        "--seed",
        "11",
    )
    assert result.returncode == 0, result.stderr
    result = _run(
        "public-run-baseline",
        "--prepared",
        str(prepared),
        "--manifest",
        str(manifest),
        "--split",
        str(split),
        "--profile",
        "span_binary",
        "--name",
        "tfidf_logistic",
        "--output-dir",
        str(output),
        "--experiment-id",
        "synthetic-baseline",
    )
    assert result.returncode == 0, result.stderr
    result = _run(
        "public-train",
        "--prepared",
        str(prepared),
        "--manifest",
        str(manifest),
        "--split",
        str(split),
        "--profile",
        "span_binary",
        "--config",
        "configs/public/smoke.yaml",
        "--output-dir",
        str(output),
        "--experiment-id",
        "synthetic-neural",
        "--device",
        "cpu",
    )
    assert result.returncode == 0, result.stderr
    result = _run(
        "public-report",
        "--result",
        str(output / "synthetic-baseline-result.json"),
        "--result",
        str(output / "synthetic-neural-result.json"),
        "--output-dir",
        str(output / "report"),
    )
    assert result.returncode == 0, result.stderr
    assert (output / "report" / "f1-comparison.svg").is_file()
