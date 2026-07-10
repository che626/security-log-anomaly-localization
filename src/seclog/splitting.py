import hashlib

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from .features import clean_log_line, nonempty_log_lines


def _template_signature(log_text: object) -> str:
    normalized = "\n".join(clean_log_line(line) for line in nonempty_log_lines(str(log_text)))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def add_template_groups(frame: pd.DataFrame) -> pd.DataFrame:
    if "log_text" not in frame.columns:
        raise ValueError("log_text is required to create template groups")
    grouped = frame.copy()
    grouped["template_group"] = grouped["log_text"].map(_template_signature)
    return grouped


def _stratification_labels(frame: pd.DataFrame) -> pd.Series:
    anomaly = frame["has_anomaly"].astype(int)
    anomaly_type = frame["primary_anomaly_type"].astype(str)
    return anomaly.astype(str) + "|" + anomaly_type.where(anomaly.eq(1), "none")


def make_locked_split(
    frame: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 20260711,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    required = {"has_anomaly", "primary_anomaly_type", "template_group"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"split frame is missing columns: {missing}")
    labels = _stratification_labels(frame)
    groups = frame["template_group"]
    n_splits = max(2, round(1 / test_size))
    label_group_counts = (
        pd.DataFrame({"label": labels, "group": groups})
        .drop_duplicates()
        .groupby("label")["group"]
        .nunique()
    )
    if len(label_group_counts) > 1 and label_group_counts.min() >= n_splits:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        tuning_indices, locked_indices = next(splitter.split(frame, labels, groups))
    else:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        )
        tuning_indices, locked_indices = next(splitter.split(frame, labels, groups))
    tuning = frame.iloc[tuning_indices].copy().reset_index(drop=True)
    locked = frame.iloc[locked_indices].copy().reset_index(drop=True)
    overlap = set(tuning["template_group"]) & set(locked["template_group"])
    if overlap:
        raise RuntimeError("template groups overlap between tuning and locked splits")
    return tuning, locked


def build_split_audit(
    frame: pd.DataFrame,
    tuning: pd.DataFrame,
    locked: pd.DataFrame,
) -> pd.DataFrame:
    sizes = frame.groupby("template_group").size().rename("template_group_size")
    membership = {
        **dict.fromkeys(tuning["id"].tolist(), "decoder_tuning"),
        **dict.fromkeys(locked["id"].tolist(), "locked_evaluation"),
    }
    audit = frame[["id", "template_group"]].copy()
    audit["template_group_size"] = audit["template_group"].map(sizes).astype(int)
    audit["is_duplicate_group"] = audit["template_group_size"].gt(1)
    audit["split"] = audit["id"].map(membership)
    if audit["split"].isna().any():
        raise ValueError("some rows are missing split membership")
    return audit
