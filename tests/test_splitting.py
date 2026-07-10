import pandas as pd

from seclog.splitting import add_template_groups, make_locked_split


def test_identical_normalized_logs_share_group() -> None:
    frame = pd.DataFrame(
        {
            "id": [1, 2],
            "log_text": [
                "2026-01-01 00:00:00 retry id=123",
                "2026-02-02 00:00:00 retry id=999",
            ],
            "has_anomaly": [1, 1],
            "primary_anomaly_type": ["timeout_retry", "timeout_retry"],
        }
    )
    grouped = add_template_groups(frame)
    assert grouped.loc[0, "template_group"] == grouped.loc[1, "template_group"]


def test_locked_split_has_disjoint_groups() -> None:
    frame = pd.DataFrame(
        {
            "id": list(range(8)),
            "has_anomaly": [0, 0, 0, 0, 1, 1, 1, 1],
            "primary_anomaly_type": [
                "none",
                "none",
                "none",
                "none",
                "timeout_retry",
                "timeout_retry",
                "resource_exhaustion",
                "resource_exhaustion",
            ],
            "template_group": [f"group-{index}" for index in range(8)],
        }
    )
    tuning, locked = make_locked_split(frame, test_size=0.25, random_state=20260711)
    assert set(tuning["template_group"]).isdisjoint(set(locked["template_group"]))
