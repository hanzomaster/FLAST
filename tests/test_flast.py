"""Unit tests for the FLAST algorithm."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import flast


class TestDataLoading:
    """Tests for data loading functions."""

    def test_get_data_points(self, temp_dataset_dir: Path) -> None:
        """Test loading data points from a directory."""
        flaky_path = temp_dataset_dir / "test-project" / "flakyMethods"
        data_points = flast.get_data_points(flaky_path)

        assert len(data_points) == 2
        assert all(isinstance(dp, str) for dp in data_points)
        assert any("flaky" in dp.lower() or "thread" in dp.lower() for dp in data_points)

    def test_get_data_points_info(self, temp_dataset_dir: Path) -> None:
        """Test loading flaky and non-flaky data points."""
        flaky, non_flaky = flast.get_data_points_info(temp_dataset_dir, "test-project")

        assert len(flaky) == 2
        assert len(non_flaky) == 3
        assert all(isinstance(dp, str) for dp in flaky + non_flaky)

    def test_legacy_get_data_points(self, temp_dataset_dir: Path) -> None:
        """Test legacy function getDataPoints."""
        flaky_path = str(temp_dataset_dir / "test-project" / "flakyMethods")
        data_points = flast.getDataPoints(flaky_path)

        assert len(data_points) == 2

    def test_legacy_get_data_points_info(self, temp_dataset_dir: Path) -> None:
        """Test legacy function getDataPointsInfo."""
        flaky, non_flaky = flast.getDataPointsInfo(str(temp_dataset_dir), "test-project")

        assert len(flaky) == 2
        assert len(non_flaky) == 3


class TestMetricsComputation:
    """Tests for metrics computation functions."""

    def test_compute_results_perfect_prediction(self) -> None:
        """Test precision/recall with perfect predictions."""
        test_labels = [1, 1, 0, 0, 0]
        predict_labels = [1, 1, 0, 0, 0]

        precision, recall = flast.compute_results(test_labels, predict_labels)

        assert precision == 1.0
        assert recall == 1.0

    def test_compute_results_all_false_positives(self) -> None:
        """Test precision with all false positives."""
        test_labels = [0, 0, 0, 0, 0]
        predict_labels = [1, 1, 1, 1, 1]

        precision, _recall = flast.compute_results(test_labels, predict_labels)

        assert precision == 0.0
        # Recall undefined when no positive samples - returns "-"

    def test_compute_results_partial_match(self) -> None:
        """Test precision/recall with partial matches."""
        test_labels = [1, 1, 1, 0, 0]
        predict_labels = [1, 0, 0, 0, 0]

        precision, recall = flast.compute_results(test_labels, predict_labels)

        assert precision == 1.0  # 1 TP, 0 FP
        assert recall == pytest.approx(1 / 3)  # 1 TP, 2 FN

    def test_legacy_compute_results(self) -> None:
        """Test legacy function computeResults."""
        test_labels = [1, 1, 0, 0]
        predict_labels = [1, 1, 0, 0]

        precision, recall = flast.computeResults(test_labels, predict_labels)

        assert precision == 1.0
        assert recall == 1.0


class TestVectorization:
    """Tests for vectorization functions."""

    def test_flast_vectorization_basic(self, sample_data_points: list[str]) -> None:
        """Test basic vectorization without projection."""
        result = flast.flast_vectorization(sample_data_points, eps=0)

        assert result.shape[0] == len(sample_data_points)
        assert result.shape[1] > 0  # Has features

    def test_flast_vectorization_with_projection(self, sample_data_points: list[str]) -> None:
        """Test vectorization with random projection."""
        # Need more samples for JL projection
        data_points = sample_data_points * 20  # 100 samples
        result = flast.flast_vectorization(data_points, eps=0.3)

        assert result.shape[0] == len(data_points)
        # With projection, dimensions should be reduced

    def test_flast_vectorization_custom_dim(self, sample_data_points: list[str]) -> None:
        """Test vectorization with custom target dimensions."""
        data_points = sample_data_points * 20
        target_dim = 10
        result = flast.flast_vectorization(data_points, dim=target_dim, eps=0.3)

        assert result.shape[0] == len(data_points)
        assert result.shape[1] == target_dim

    def test_legacy_flast_vectorization(self, sample_data_points: list[str]) -> None:
        """Test legacy function flastVectorization."""
        result = flast.flastVectorization(sample_data_points, eps=0)

        assert result.shape[0] == len(sample_data_points)


class TestClassification:
    """Tests for classification functions."""

    def test_flast_classification_basic(
        self,
        sample_train_data: np.ndarray,
        sample_test_data: np.ndarray,
        sample_train_labels: list[int],
        knn_params: dict[str, str],
    ) -> None:
        """Test basic classification."""
        train_time, test_time, predictions = flast.flast_classification(
            sample_train_data,
            sample_train_labels,
            sample_test_data,
            sigma=0.5,
            k=3,
            params=knn_params,
        )

        assert train_time >= 0
        assert test_time >= 0
        assert len(predictions) == sample_test_data.shape[0]
        assert all(p in [0, 1] for p in predictions)

    def test_flast_classification_different_k(
        self,
        sample_train_data: np.ndarray,
        sample_test_data: np.ndarray,
        sample_train_labels: list[int],
        knn_params: dict[str, str],
    ) -> None:
        """Test classification with different k values."""
        for k in [1, 3, 5, 7]:
            _, _, predictions = flast.flast_classification(
                sample_train_data,
                sample_train_labels,
                sample_test_data,
                sigma=0.5,
                k=k,
                params=knn_params,
            )

            assert len(predictions) == sample_test_data.shape[0]

    def test_flast_classification_different_sigma(
        self,
        sample_train_data: np.ndarray,
        sample_test_data: np.ndarray,
        sample_train_labels: list[int],
        knn_params: dict[str, str],
    ) -> None:
        """Test classification with different sigma thresholds."""
        for sigma in [0.1, 0.5, 0.9]:
            _, _, predictions = flast.flast_classification(
                sample_train_data,
                sample_train_labels,
                sample_test_data,
                sigma=sigma,
                k=3,
                params=knn_params,
            )

            assert len(predictions) == sample_test_data.shape[0]

    def test_flast_classification_uniform_weights(
        self,
        sample_train_data: np.ndarray,
        sample_test_data: np.ndarray,
        sample_train_labels: list[int],
    ) -> None:
        """Test classification with uniform weights."""
        params = {
            "algorithm": "brute",
            "metric": "cosine",
            "weights": "uniform",
        }
        _, _, predictions = flast.flast_classification(
            sample_train_data,
            sample_train_labels,
            sample_test_data,
            sigma=0.5,
            k=3,
            params=params,
        )

        assert len(predictions) == sample_test_data.shape[0]

    def test_legacy_flast_classification(
        self,
        sample_train_data: np.ndarray,
        sample_test_data: np.ndarray,
        sample_train_labels: list[int],
        knn_params: dict[str, str],
    ) -> None:
        """Test legacy function flastClassification."""
        _train_time, _test_time, predictions = flast.flastClassification(
            sample_train_data,
            sample_train_labels,
            sample_test_data,
            sigma=0.5,
            k=3,
            params=knn_params,
        )

        assert len(predictions) == sample_test_data.shape[0]


class TestComputePrediction:
    """Tests for the _compute_prediction helper function."""

    def test_prediction_high_phi(self) -> None:
        """Test prediction when phi dominates."""
        result = flast._compute_prediction(phi=0.8, psi=0.2, sigma=0.5)
        assert result == 1

    def test_prediction_high_psi(self) -> None:
        """Test prediction when psi dominates."""
        result = flast._compute_prediction(phi=0.2, psi=0.8, sigma=0.5)
        assert result == 0

    def test_prediction_at_threshold(self) -> None:
        """Test prediction at exact threshold."""
        result = flast._compute_prediction(phi=0.5, psi=0.5, sigma=0.5)
        assert result == 1  # phi/(phi+psi) = 0.5 >= sigma

    def test_prediction_below_threshold(self) -> None:
        """Test prediction below threshold."""
        result = flast._compute_prediction(phi=0.4, psi=0.6, sigma=0.5)
        assert result == 0

    def test_prediction_both_zero(self) -> None:
        """Test prediction when both phi and psi are zero."""
        result = flast._compute_prediction(phi=0.0, psi=0.0, sigma=0.5)
        assert result == 0

    def test_prediction_phi_infinite(self) -> None:
        """Test prediction when phi is infinite."""
        result = flast._compute_prediction(phi=float("inf"), psi=1.0, sigma=0.5)
        assert result == 1

    def test_prediction_psi_infinite(self) -> None:
        """Test prediction when psi is infinite."""
        result = flast._compute_prediction(phi=1.0, psi=float("inf"), sigma=0.5)
        assert result == 0

    def test_prediction_both_infinite(self) -> None:
        """Test prediction when both are infinite."""
        result = flast._compute_prediction(phi=float("inf"), psi=float("inf"), sigma=0.5)
        assert result == 0


class TestIntegration:
    """Integration tests for the full FLAST pipeline."""

    def test_full_pipeline(self, temp_dataset_dir: Path) -> None:
        """Test the full FLAST pipeline from data loading to prediction."""
        # Load data
        flaky, non_flaky = flast.get_data_points_info(temp_dataset_dir, "test-project")
        all_data = flaky + non_flaky
        labels = [1] * len(flaky) + [0] * len(non_flaky)

        # Vectorize
        vectors = flast.flast_vectorization(all_data, eps=0)

        # Convert to numpy arrays
        data_array = np.array([vectors[i].toarray() for i in range(vectors.shape[0])])
        n_samples, nx, ny = data_array.shape
        data_array = data_array.reshape((n_samples, nx * ny))

        # Split into train/test
        train_data = data_array[:3]
        test_data = data_array[3:]
        train_labels = labels[:3]
        test_labels = labels[3:]

        # Classify
        params = {"algorithm": "brute", "metric": "cosine", "weights": "distance"}
        _, _, predictions = flast.flast_classification(
            train_data, train_labels, test_data, sigma=0.5, k=2, params=params
        )

        # Compute metrics
        _precision, _recall = flast.compute_results(test_labels, predictions)

        # Basic sanity checks
        assert len(predictions) == len(test_labels)
        assert all(p in [0, 1] for p in predictions)
