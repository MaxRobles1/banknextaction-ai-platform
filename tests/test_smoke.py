"""Smoke test: the package installs and exposes its version."""

import banknextaction


def test_package_exposes_version() -> None:
    assert isinstance(banknextaction.__version__, str)
    assert banknextaction.__version__.count(".") == 2
