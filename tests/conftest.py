"""Pytest configuration and fixtures for FLAST tests."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def sample_data_points() -> list[str]:
    """Sample tokenized test method data."""
    return [
        "public void test method assert equals expected actual",
        "test async wait sleep thread timeout exception",
        "public void verify data load database connection",
        "test mock service response http client",
        "assert true condition check validation",
    ]


@pytest.fixture
def sample_labels() -> list[int]:
    """Sample labels (1 = flaky, 0 = non-flaky)."""
    return [1, 1, 0, 0, 0]


@pytest.fixture
def temp_dataset_dir() -> Generator[Path, None, None]:
    """Create a temporary dataset directory with sample data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir) / "test-project"
        flaky_path = project_path / "flakyMethods"
        non_flaky_path = project_path / "nonFlakyMethods"

        flaky_path.mkdir(parents=True)
        non_flaky_path.mkdir(parents=True)

        # Create sample flaky test files
        flaky_tests = [
            "public void testFlaky1 thread sleep random timeout",
            "async await promise reject timeout flaky",
        ]
        for i, content in enumerate(flaky_tests):
            (flaky_path / f"test_{i}.txt").write_text(content)

        # Create sample non-flaky test files
        non_flaky_tests = [
            "public void testStable1 assert equals expected",
            "test verify data consistent result",
            "assert true condition check",
        ]
        for i, content in enumerate(non_flaky_tests):
            (non_flaky_path / f"test_{i}.txt").write_text(content)

        yield Path(tmpdir)


@pytest.fixture
def sample_train_data() -> np.ndarray:
    """Sample vectorized training data."""
    return np.random.rand(10, 50).astype(np.float64)


@pytest.fixture
def sample_test_data() -> np.ndarray:
    """Sample vectorized test data."""
    return np.random.rand(3, 50).astype(np.float64)


@pytest.fixture
def sample_train_labels() -> list[int]:
    """Sample training labels."""
    return [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]


@pytest.fixture
def knn_params() -> dict[str, str]:
    """Default k-NN parameters."""
    return {
        "algorithm": "brute",
        "metric": "cosine",
        "weights": "distance",
    }
