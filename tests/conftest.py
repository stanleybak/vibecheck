"""Shared pytest fixtures and config."""

import os
import yaml
import pytest
from pathlib import Path

PATHS_FILE = Path(__file__).parent / "paths.yaml"


def _load_paths():
    if not PATHS_FILE.exists():
        return {}
    with open(PATHS_FILE) as f:
        return yaml.safe_load(f) or {}


@pytest.fixture(scope="session")
def paths():
    """Load external paths from tests/paths.yaml."""
    return _load_paths()


@pytest.fixture(scope="session")
def vnncomp_benchmarks(paths):
    p = paths.get("vnncomp_benchmarks")
    if not p or not Path(p).exists():
        pytest.skip("vnncomp_benchmarks path not configured in tests/paths.yaml")
    return Path(p)


@pytest.fixture(scope="session")
def vnncomp_results(paths):
    p = paths.get("vnncomp_results")
    if not p or not Path(p).exists():
        pytest.skip("vnncomp_results path not configured in tests/paths.yaml")
    return Path(p)
