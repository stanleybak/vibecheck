"""Shared pytest fixtures and config."""

import os
import yaml
import pytest
from pathlib import Path

PATHS_FILE = Path(__file__).parent / "paths.yaml"
PATHS_TEMPLATE = Path(__file__).parent / "paths.yaml.template"


def _load_paths():
    if not PATHS_FILE.exists():
        pytest.fail(
            f"Missing {PATHS_FILE}\n\n"
            f"Setup:\n"
            f"  1. Clone benchmarks: git clone https://github.com/stanleybak/vnncomp2025_benchmarks.git\n"
            f"  2. Copy template:    cp {PATHS_TEMPLATE} {PATHS_FILE}\n"
            f"  3. Edit paths.yaml with your local benchmark path\n"
        )
    with open(PATHS_FILE) as f:
        return yaml.safe_load(f) or {}


@pytest.fixture(scope="session")
def paths():
    """Load external paths from tests/paths.yaml."""
    return _load_paths()


@pytest.fixture(scope="session")
def vnncomp_benchmarks(paths):
    p = paths.get("vnncomp_benchmarks")
    if not p:
        pytest.fail(
            f"'vnncomp_benchmarks' not set in {PATHS_FILE}\n"
            f"Add:  vnncomp_benchmarks: /path/to/vnncomp2025_benchmarks"
        )
    path = Path(p)
    if not path.exists():
        pytest.fail(
            f"vnncomp_benchmarks path does not exist: {p}\n\n"
            f"Clone it:\n"
            f"  git clone https://github.com/stanleybak/vnncomp2025_benchmarks.git\n"
            f"Then update {PATHS_FILE}"
        )
    # Auto-resolve benchmarks/ subfolder
    if (path / "benchmarks").is_dir():
        path = path / "benchmarks"
    return path


@pytest.fixture(scope="session")
def vnncomp_results(paths):
    p = paths.get("vnncomp_results")
    if not p:
        pytest.fail(f"'vnncomp_results' not set in {PATHS_FILE}")
    path = Path(p)
    if not path.exists():
        pytest.fail(f"vnncomp_results path does not exist: {p}")
    return path
