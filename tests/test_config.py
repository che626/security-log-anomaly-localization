from seclog.config import load_config


def test_smoke_config_is_cpu_sized() -> None:
    config = load_config("configs/smoke.yaml")
    assert config.model.hidden == 16
    assert config.training.folds == 2
    assert config.training.epochs == 1


def test_final_config_matches_selected_run() -> None:
    config = load_config("configs/final.yaml")
    assert config.model.emb_dim == 128
    assert config.model.hidden == 176
    assert config.training.seed == 999
    assert config.training.folds == 5
