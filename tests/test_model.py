import torch

from seclog.data import Sample, pack_log_batch
from seclog.model import LogBoundaryNetwork


def make_batch() -> dict:
    return pack_log_batch(
        [
            Sample(sid=1, lines=["a", "b"], token_ids=[[1], [2]]),
            Sample(sid=2, lines=["c"], token_ids=[[3]]),
        ]
    )


def test_model_output_shapes() -> None:
    batch = make_batch()
    model = LogBoundaryNetwork(vocab_size=64, emb_dim=8, hidden=12, num_layers=1, dropout=0)
    tag, start, end, global_logits = model(
        batch["input_ids"],
        batch["offsets"],
        batch["owner"],
        batch["mask"],
        batch["pos_feats"],
    )
    assert tuple(tag.shape) == (2, 2, 21)
    assert tuple(start.shape) == (2, 2, 10)
    assert tuple(end.shape) == (2, 2, 10)
    assert tuple(global_logits.shape) == (2, 11)


def test_state_dict_round_trip(tmp_path) -> None:
    torch.manual_seed(7)
    original = LogBoundaryNetwork(64, 8, 12, 1, 0)
    path = tmp_path / "model.pt"
    torch.save(original.state_dict(), path)
    restored = LogBoundaryNetwork(64, 8, 12, 1, 0)
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    for left, right in zip(original.parameters(), restored.parameters()):
        assert torch.equal(left, right)
