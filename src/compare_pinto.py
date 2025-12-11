"""RQ3: Comparison with Pinto-KNN approach.

This script compares FLAST against the Pinto-KNN baseline approach
for test flakiness prediction.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

import flast

if TYPE_CHECKING:
    from numpy.typing import NDArray


def flast_knn(
    out_dir: Path,
    project_base_path: Path,
    project_name: str,
    kf: ShuffleSplit,
    dim: int,
    eps: float,
    k: int,
    sigma: float,
    params: dict[str, Any],
) -> tuple[float | str, float, int, float, float]:
    """Run FLAST k-NN evaluation with cross-validation.

    Args:
        out_dir: Output directory for temporary files.
        project_base_path: Base directory containing project datasets.
        project_name: Name of the project to evaluate.
        kf: Cross-validation splitter.
        dim: Target dimensionality for projection.
        eps: Johnson-Lindenstrauss error tolerance.
        k: Number of neighbors.
        sigma: Decision threshold.
        params: k-NN classifier parameters.

    Returns:
        Tuple of (avg_precision, avg_recall, storage, avg_prep_time, avg_pred_time).
    """
    v0 = time.perf_counter()
    data_points_flaky, data_points_non_flaky = flast.getDataPointsInfo(
        str(project_base_path), project_name
    )
    data_points = data_points_flaky + data_points_non_flaky
    z = flast.flastVectorization(data_points, dim=dim, eps=eps)
    data_points_list: NDArray[Any] = np.array([z[i].toarray() for i in range(z.shape[0])])
    data_labels_list: NDArray[Any] = np.array(
        [1] * len(data_points_flaky) + [0] * len(data_points_non_flaky)
    )
    v1 = time.perf_counter()
    vec_time = v1 - v0

    # storage
    knn_data = (data_points_list, data_labels_list)
    pickle_dump_path = out_dir / f"flast-k{k}-sigma{sigma}.pickle"
    with pickle_dump_path.open("wb") as pickle_file:
        pickle.dump(knn_data, pickle_file)
    storage = pickle_dump_path.stat().st_size
    pickle_dump_path.unlink()

    avg_p: float | str = 0.0
    avg_r = 0.0
    avg_t_prep, avg_t_pred = 0.0, 0.0
    success_fold, precision_fold = 0, 0

    for trn_idx, tst_idx in kf.split(data_points_list, data_labels_list):
        train_data, test_data = data_points_list[trn_idx], data_points_list[tst_idx]
        train_labels, test_labels = data_labels_list[trn_idx], data_labels_list[tst_idx]
        if sum(train_labels) == 0 or sum(test_labels) == 0:
            print("Skipping fold...")
            print(" Flaky Train Tests", sum(train_labels))
            print(" Flaky Test Tests", sum(test_labels))
            continue

        success_fold += 1

        # prepare the data in the right format for kNN
        n_samples_train, nx_train, ny_train = train_data.shape
        train_data = train_data.reshape((n_samples_train, nx_train * ny_train))
        n_samples_test, nx_test, ny_test = test_data.shape
        test_data = test_data.reshape((n_samples_test, nx_test * ny_test))

        train_time, test_time, predict_labels = flast.flastClassification(
            train_data, train_labels.tolist(), test_data, sigma, k, params
        )

        preparation_time = (vec_time * len(train_data) / len(data_points)) + train_time
        prediction_time = (vec_time / len(data_points)) + (test_time / len(test_data))
        precision, recall = flast.computeResults(test_labels.tolist(), predict_labels)
        print(precision, recall)
        if precision != "-":
            precision_fold += 1
            avg_p += precision  # type: ignore[operator]
        avg_r += recall  # type: ignore[operator]
        avg_t_prep += preparation_time
        avg_t_pred += prediction_time

    if precision_fold == 0:
        avg_p = "-"
    else:
        avg_p /= precision_fold  # type: ignore[operator]
    avg_r /= success_fold
    avg_t_prep /= success_fold
    avg_t_pred /= success_fold

    return avg_p, avg_r, storage, avg_t_prep, avg_t_pred


def pinto_knn(
    out_dir: Path,
    project_base_path: Path,
    project_name: str,
    kf: ShuffleSplit,
) -> tuple[float | str, float, int, float, float]:
    """Run Pinto-KNN baseline evaluation with cross-validation.

    Args:
        out_dir: Output directory for temporary files.
        project_base_path: Base directory containing project datasets.
        project_name: Name of the project to evaluate.
        kf: Cross-validation splitter.

    Returns:
        Tuple of (avg_precision, avg_recall, storage, avg_prep_time, avg_pred_time).
    """
    v0 = time.perf_counter()
    data_points_flaky, data_points_non_flaky = flast.getDataPointsInfo(
        str(project_base_path), project_name
    )
    data_points = data_points_flaky + data_points_non_flaky
    count_vec = CountVectorizer()
    z = count_vec.fit_transform(data_points)
    data_points_list: NDArray[Any] = np.array([z[i].toarray() for i in range(z.shape[0])])
    data_labels_list: NDArray[Any] = np.array(
        [1] * len(data_points_flaky) + [0] * len(data_points_non_flaky)
    )
    v1 = time.perf_counter()
    vec_time = v1 - v0

    # storage
    knn_data = (data_points_list, data_labels_list)
    pickle_dump_path = out_dir / "PintoKNN.pickle"
    with pickle_dump_path.open("wb") as pickle_file:
        pickle.dump(knn_data, pickle_file)
    storage = pickle_dump_path.stat().st_size
    pickle_dump_path.unlink()

    avg_p: float | str = 0.0
    avg_r = 0.0
    avg_t_prep, avg_t_pred = 0.0, 0.0
    success_fold, precision_fold = 0, 0

    for trn_idx, tst_idx in kf.split(data_points_list, data_labels_list):
        train_data, test_data = data_points_list[trn_idx], data_points_list[tst_idx]
        train_labels, test_labels = data_labels_list[trn_idx], data_labels_list[tst_idx]
        if sum(train_labels) == 0 or sum(test_labels) == 0:
            print("Skipping fold...")
            print(" Flaky Train Tests", sum(train_labels))
            print(" Flaky Test Tests", sum(test_labels))
            continue

        success_fold += 1

        # prepare the data in the right format for kNN
        n_samples_train, nx_train, ny_train = train_data.shape
        train_data = train_data.reshape((n_samples_train, nx_train * ny_train))
        n_samples_test, nx_test, ny_test = test_data.shape
        test_data = test_data.reshape((n_samples_test, nx_test * ny_test))

        # training
        t0 = time.perf_counter()
        knn = KNeighborsClassifier(
            algorithm="brute",
            metric="euclidean",
            weights="uniform",
            n_neighbors=1,
            n_jobs=1,
        )
        knn.fit(train_data, train_labels)
        t1 = time.perf_counter()
        train_time = t1 - t0

        # testing
        p0 = time.perf_counter()
        predict_labels = knn.predict(test_data)
        p1 = time.perf_counter()
        test_time = p1 - p0

        preparation_time = (vec_time * len(train_data) / len(data_points)) + train_time
        prediction_time = (vec_time / len(data_points)) + (test_time / len(test_data))
        precision, recall = flast.computeResults(test_labels.tolist(), predict_labels.tolist())
        print(precision, recall)
        if precision != "-":
            precision_fold += 1
            avg_p += precision  # type: ignore[operator]
        avg_r += recall  # type: ignore[operator]
        avg_t_prep += preparation_time
        avg_t_pred += prediction_time

    if precision_fold == 0:
        avg_p = "-"
    else:
        avg_p /= precision_fold  # type: ignore[operator]
    avg_r /= success_fold
    avg_t_prep /= success_fold
    avg_t_pred /= success_fold

    return avg_p, avg_r, storage, avg_t_prep, avg_t_pred


# Legacy function names for backward compatibility
flastKNN = flast_knn
pintoKNN = pinto_knn


if __name__ == "__main__":
    project_base_path = Path("dataset")
    project_name = "pinto-ds"

    out_dir = Path("results")
    out_file = "compare-pinto.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / out_file).open("w") as fo:
        fo.write("approach,precision,recall,storage,preparationTime,predictionTime\n")

    # CV
    num_split = 30
    test_size = 0.2
    kf = ShuffleSplit(n_splits=num_split, test_size=test_size)

    # PINTO
    print("Pinto-KNN")
    avg_p, avg_r, storage, avg_t_prep, avg_t_pred = pinto_knn(
        out_dir, project_base_path, project_name, kf
    )
    with (out_dir / out_file).open("a") as fo:
        fo.write(f"Pinto-KNN,{avg_p},{avg_r},{storage},{avg_t_prep},{avg_t_pred}\n")

    # FLAST
    dim = 0  # number of dimensions (0: JL with error eps)
    eps = 0.3  # JL eps
    params: dict[str, Any] = {
        "algorithm": "brute",
        "metric": "cosine",
        "weights": "distance",
    }

    for k in [3, 7]:
        for sigma in [0.5, 0.95]:
            print("FLAST", k, sigma)
            avg_p, avg_r, storage, avg_t_prep, avg_t_pred = flast_knn(
                out_dir, project_base_path, project_name, kf, dim, eps, k, sigma, params
            )
            with (out_dir / out_file).open("a") as fo:
                fo.write(
                    f"FLAST-k{k}-sigma{sigma},{avg_p},{avg_r},{storage},{avg_t_prep},{avg_t_pred}\n"
                )
