import re

ANOMALY_TYPES = (
    "timeout_retry",
    "resource_exhaustion",
    "slow_burn_warning",
    "state_conflict",
    "parameter_drift",
    "out_of_order",
    "missing_step",
    "duplicate_event",
    "cross_component_mismatch",
    "partial_recovery_loop",
)
TYPE_TO_ID = {name: index for index, name in enumerate(ANOMALY_TYPES)}
N_TYPES = len(ANOMALY_TYPES)
N_LABELS = 1 + 2 * N_TYPES
GLOBAL_NONE_ID = N_TYPES
SUBMISSION_COLUMNS = (
    "id",
    "has_anomaly",
    "primary_start_idx",
    "primary_end_idx",
    "primary_anomaly_type",
    "all_spans",
)
TRAIN_COLUMNS = ("id", "log_text", *SUBMISSION_COLUMNS[1:])
TEST_COLUMNS = ("id", "log_text")

TS_RE = re.compile(r"^\s*\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s*")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
HEX_RE = re.compile(r"0x[0-9a-fA-F]+")
NUM_RE = re.compile(r"(?<![a-zA-Z_])[-+]?\d+(?:\.\d+)?")
SEG_RE = re.compile(r"\bseg[_-]?[a-zA-Z0-9]+\b", re.I)
PATH_RE = re.compile(r"/(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]*")
WORD_RE = re.compile(r"[a-zA-Z_<>][a-zA-Z0-9_<>-]*|[=:/.-]+")
