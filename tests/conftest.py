"""Pytest configuration and shared fixtures for track generation tool tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add src directory to path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (may require external resources)",
    )


@pytest.fixture(scope="session")
def test_data_directory() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def package_data_directory() -> Path:
    """Return path to package data directory with model files."""
    return Path(__file__).parent.parent / "src" / "cam_track_gen" / "data"
