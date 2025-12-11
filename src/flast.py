"""FLAST: Fast Static Prediction of Test Flakiness.

This module implements the FLAST algorithm for predicting test flakiness
using k-Nearest Neighbors classification on tokenized test methods.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.random_projection import (
    SparseRandomProjection,
    johnson_lindenstrauss_min_dim,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.sparse import spmatrix


###############################################################################
# Data Loading


def get_data_points(path: Path) -> list[str]:
    """Read tokenized test methods from a directory.

    Args:
        path: Directory containing tokenized test method files.

    Returns:
        List of tokenized test methods as strings.
    """
    data_points: list[str] = []
    for data_point_file in path.iterdir():
        if data_point_file.name.startswith("."):
            continue
        data_points.append(data_point_file.read_text(encoding="utf-8"))
    return data_points


def get_data_points_info(
    project_base_path: Path, project_name: str
) -> tuple[list[str], list[str]]:
    """Get flaky and non-flaky test methods for a project.

    Args:
        project_base_path: Base directory containing project datasets.
        project_name: Name of the project to load.

    Returns:
        Tuple of (flaky_methods, non_flaky_methods) as lists of strings.
    """
    project_path = project_base_path / project_name
    flaky_path = project_path / "flakyMethods"
    non_flaky_path = project_path / "nonFlakyMethods"
    return get_data_points(flaky_path), get_data_points(non_flaky_path)


# Legacy function names for backward compatibility
def getDataPoints(path: str) -> list[str]:
    """Legacy wrapper for get_data_points.

    Deprecated: Use get_data_points() instead.
    """
    return get_data_points(Path(path))


def getDataPointsInfo(
    projectBasePath: str, projectName: str
) -> tuple[list[str], list[str]]:
    """Legacy wrapper for get_data_points_info.

    Deprecated: Use get_data_points_info() instead.
    """
    return get_data_points_info(Path(projectBasePath), projectName)


###############################################################################
# Metrics Computation


def compute_results(
    test_labels: list[int], predict_labels: list[int]
) -> tuple[float | str, float | str]:
    """Compute precision and recall for predictions.

    Args:
        test_labels: Ground truth labels (1 for flaky, 0 for non-flaky).
        predict_labels: Predicted labels.

    Returns:
        Tuple of (precision, recall). Returns "-" if metric cannot be computed.
    """
    precision: float | str
    recall: float | str

    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try:
            precision = float(precision_score(test_labels, predict_labels))
        except (ValueError, Warning):
            precision = "-"
        try:
            recall = float(recall_score(test_labels, predict_labels))
        except (ValueError, Warning):
            recall = "-"

    return precision, recall


# Legacy function name for backward compatibility
def computeResults(
    testLabels: list[int], predictLabels: list[int]
) -> tuple[float | str, float | str]:
    """Legacy wrapper for compute_results.

    Deprecated: Use compute_results() instead.
    """
    return compute_results(testLabels, predictLabels)


###############################################################################
# FLAST Algorithm


def flast_vectorization(
    data_points: list[str],
    dim: int = 0,
    eps: float = 0.3,
) -> NDArray[Any] | spmatrix:
    """Vectorize text data using CountVectorizer and optional random projection.

    Uses the bag-of-words representation followed by optional dimensionality
    reduction via Johnson-Lindenstrauss random projection.

    Args:
        data_points: List of tokenized test methods.
        dim: Target dimensionality for projection. If <= 0, computed automatically.
        eps: Error tolerance for Johnson-Lindenstrauss projection.
            Set to 0 to disable projection.

    Returns:
        Vectorized representation of the input data.
    """
    count_vec = CountVectorizer()
    z_full = count_vec.fit_transform(data_points)

    if eps == 0:
        return z_full

    if dim <= 0:
        dim = johnson_lindenstrauss_min_dim(z_full.shape[0], eps=eps)

    srp = SparseRandomProjection(n_components=dim)
    return srp.fit_transform(z_full)


def flast_classification(
    train_data: NDArray[Any] | spmatrix,
    train_labels: list[int],
    test_data: NDArray[Any] | spmatrix,
    sigma: float,
    k: int,
    params: dict[str, Any],
) -> tuple[float, float, list[int]]:
    """Classify test data using k-NN with custom voting.

    Args:
        train_data: Vectorized training data.
        train_labels: Training labels (1 for flaky, 0 for non-flaky).
        test_data: Vectorized test data.
        sigma: Decision threshold for flaky classification.
        k: Number of neighbors to consider.
        params: k-NN parameters (algorithm, metric, weights).

    Returns:
        Tuple of (train_time, test_time, predicted_labels).
    """
    # Training
    t0 = time.perf_counter()
    knn = KNeighborsClassifier(
        algorithm=params["algorithm"],
        metric=params["metric"],
        weights=params["weights"],
        n_neighbors=k,
        n_jobs=1,
    )
    knn.fit(train_data, train_labels)
    t1 = time.perf_counter()
    train_time = t1 - t0

    # Prediction with custom voting
    t0 = time.perf_counter()
    predict_labels: list[int] = []
    neighbor_dist, neighbor_ind = knn.kneighbors(test_data)

    use_distance_weights = knn.get_params()["weights"] == "distance"

    for distances, indices in zip(neighbor_dist, neighbor_ind, strict=True):
        phi, psi = 0.0, 0.0

        for distance, neighbor in zip(distances, indices, strict=True):
            if use_distance_weights:
                d_inv = (1 / distance) if distance != 0 else float("inf")
            else:
                d_inv = 1.0

            if train_labels[neighbor] == 1:
                phi += d_inv
            else:
                psi += d_inv

        # Handle edge cases for prediction
        prediction = _compute_prediction(phi, psi, sigma)
        predict_labels.append(prediction)

    t1 = time.perf_counter()
    test_time = t1 - t0

    return train_time, test_time, predict_labels


def _compute_prediction(phi: float, psi: float, sigma: float) -> int:
    """Compute prediction based on weighted neighbor votes.

    Args:
        phi: Weighted sum of flaky neighbor votes.
        psi: Weighted sum of non-flaky neighbor votes.
        sigma: Decision threshold.

    Returns:
        1 if predicted flaky, 0 otherwise.
    """
    inf = float("inf")

    if phi == inf and psi == inf:
        return 0
    if psi == inf:
        return 0
    if phi == inf:
        return 1
    if (phi + psi) == 0:
        return 0
    if phi / (phi + psi) >= sigma:
        return 1
    return 0


# Legacy function names for backward compatibility
def flastVectorization(
    dataPoints: list[str],
    dim: int = 0,
    eps: float = 0.3,
) -> NDArray[Any] | spmatrix:
    """Legacy wrapper for flast_vectorization.

    Deprecated: Use flast_vectorization() instead.
    """
    return flast_vectorization(dataPoints, dim, eps)


def flastClassification(
    trainData: NDArray[Any] | spmatrix,
    trainLabels: list[int],
    testData: NDArray[Any] | spmatrix,
    sigma: float,
    k: int,
    params: dict[str, Any],
) -> tuple[float, float, list[int]]:
    """Legacy wrapper for flast_classification.

    Deprecated: Use flast_classification() instead.
    """
    return flast_classification(trainData, trainLabels, testData, sigma, k, params)
