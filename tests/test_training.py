from pathlib import Path

from seclog.config import load_config
from seclog.training import run_training


def test_cpu_smoke_training_writes_two_folds(tmp_path: Path) -> None:
    result = run_training(
        data_dir=Path("tests/fixtures/synthetic_data"),
        output_dir=tmp_path,
        config=load_config("configs/smoke.yaml"),
        device_name="cpu",
    )
    assert len(result.checkpoints) == 2
    assert result.oof_path.exists()
    assert result.metrics_path.exists()
