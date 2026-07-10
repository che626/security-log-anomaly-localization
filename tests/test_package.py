from importlib.metadata import version

import seclog


def test_package_exposes_version() -> None:
    assert seclog.__version__ == version("security-log-anomaly-localization")
