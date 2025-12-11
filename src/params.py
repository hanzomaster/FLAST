"""Parameter tuning experiments for FLAST.

This script evaluates the effect of different FLAST parameters
(distance metric, epsilon, k neighbors, sigma threshold, training set size)
on prediction performance.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

import flast

if TYPE_CHECKING:
    from numpy.typing import NDArray


def flast_knn(
    out_dir: Path,
    project_base_path: Path,
    project_name: str,
    kf: StratifiedShuffleSplit,
    dim: int,
    eps: float,
    k: int,
    sigma: float,
    params: dict[str, Any],
) -> tuple[float, float, float, float, float | str, float, int, float, float]:
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
        Tuple of metrics: (avg_flaky_train, avg_non_flaky_train, avg_flaky_test,
        avg_non_flaky_test, avg_precision, avg_recall, storage, avg_prep_time,
        avg_pred_time).
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
    avg_flaky_train, avg_non_flaky_train = 0.0, 0.0
    avg_flaky_test, avg_non_flaky_test = 0.0, 0.0
    success_fold, precision_fold = 0, 0

    for trn_idx, tst_idx in kf.split(data_points_list, data_labels_list):
        # Re-vectorize for each fold (as in original implementation)
        data_points_flaky, data_points_non_flaky = flast.getDataPointsInfo(
            str(project_base_path), project_name
        )
        data_points = data_points_flaky + data_points_non_flaky
        z = flast.flastVectorization(data_points, dim=dim, eps=eps)
        data_points_list = np.array([z[i].toarray() for i in range(z.shape[0])])
        data_labels_list = np.array([1] * len(data_points_flaky) + [0] * len(data_points_non_flaky))

        train_data, test_data = data_points_list[trn_idx], data_points_list[tst_idx]
        train_labels, test_labels = data_labels_list[trn_idx], data_labels_list[tst_idx]
        if sum(train_labels) == 0 or sum(test_labels) == 0:
            print("Skipping fold...")
            print(" Flaky Train Tests", sum(train_labels))
            print(" Flaky Test Tests", sum(test_labels))
            continue

        success_fold += 1
        avg_flaky_train += sum(train_labels)
        avg_non_flaky_train += len(train_labels) - sum(train_labels)
        avg_flaky_test += sum(test_labels)
        avg_non_flaky_test += len(test_labels) - sum(test_labels)

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
    avg_flaky_train /= success_fold
    avg_non_flaky_train /= success_fold
    avg_flaky_test /= success_fold
    avg_non_flaky_test /= success_fold

    return (
        avg_flaky_train,
        avg_non_flaky_train,
        avg_flaky_test,
        avg_non_flaky_test,
        avg_p,
        avg_r,
        storage,
        avg_t_prep,
        avg_t_pred,
    )


# Legacy function name for backward compatibility
flastKNN = flast_knn


if __name__ == "__main__":
    project_base_path = Path("dataset")
    project_name = "pinto-ds"
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    num_split = 30
    test_set_size = 0.2
    kf = StratifiedShuffleSplit(n_splits=num_split, test_size=test_set_size)

    # DISTANCE
    out_file = "params-distance.csv"
    with (out_dir / out_file).open("w") as fo:
        fo.write("distance,k,sigma,eps,precision,recall,storage,preparationTime,predictionTime\n")

    k = 7
    sigma = 0.5
    dim = 0  # number of dimensions (0: JL with error eps)
    eps = 0.3  # JL eps
    params: dict[str, Any] = {
        "algorithm": "brute",
        "metric": "cosine",
        "weights": "uniform",
    }

    for metric in ["cosine", "euclidean"]:
        for k in [3, 7]:
            print(f"{metric=}, {k=}")
            params["metric"] = metric
            (
                _flaky_train,
                _non_flaky_train,
                _flaky_test,
                _non_flaky_test,
                avg_p,
                avg_r,
                storage,
                avg_t_prep,
                avg_t_pred,
            ) = flast_knn(out_dir, project_base_path, project_name, kf, dim, eps, k, sigma, params)
            with (out_dir / out_file).open("a") as fo:
                fo.write(
                    f"{params['metric']},{k},{sigma},{eps},{avg_p},{avg_r},"
                    f"{storage},{avg_t_prep},{avg_t_pred}\n"
                )

    # EPSILON
    out_file = "params-eps.csv"
    with (out_dir / out_file).open("w") as fo:
        fo.write("distance,k,sigma,eps,precision,recall,storage,preparationTime,predictionTime\n")

    k = 7
    sigma = 0.5
    dim = 0
    eps = 0.3
    params = {"algorithm": "brute", "metric": "cosine", "weights": "uniform"}

    for eps in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        print(f"{eps=}")
        (
            _flaky_train,
            _non_flaky_train,
            _flaky_test,
            _non_flaky_test,
            avg_p,
            avg_r,
            storage,
            avg_t_prep,
            avg_t_pred,
        ) = flast_knn(out_dir, project_base_path, project_name, kf, dim, eps, k, sigma, params)
        with (out_dir / out_file).open("a") as fo:
            fo.write(
                f"{params['metric']},{k},{sigma},{eps},{avg_p},{avg_r},"
                f"{storage},{avg_t_prep},{avg_t_pred}\n"
            )

    # NEIGHBORS K
    out_file = "params-k.csv"
    with (out_dir / out_file).open("w") as fo:
        fo.write("distance,k,sigma,eps,precision,recall,storage,preparationTime,predictionTime\n")

    k = 7
    sigma = 0.5
    dim = 0
    eps = 0.3
    params = {"algorithm": "brute", "metric": "cosine", "weights": "uniform"}

    for k in [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]:
        print(f"{k=}")
        (
            _flaky_train,
            _non_flaky_train,
            _flaky_test,
            _non_flaky_test,
            avg_p,
            avg_r,
            storage,
            avg_t_prep,
            avg_t_pred,
        ) = flast_knn(out_dir, project_base_path, project_name, kf, dim, eps, k, sigma, params)
        with (out_dir / out_file).open("a") as fo:
            fo.write(
                f"{params['metric']},{k},{sigma},{eps},{avg_p},{avg_r},"
                f"{storage},{avg_t_prep},{avg_t_pred}\n"
            )

    # THRESHOLD SIGMA
    out_file = "params-sigma.csv"
    with (out_dir / out_file).open("w") as fo:
        fo.write("distance,k,sigma,eps,precision,recall,storage,preparationTime,predictionTime\n")

    k = 7
    sigma = 0.5
    dim = 0
    eps = 0.3
    params = {"algorithm": "brute", "metric": "cosine", "weights": "uniform"}

    sigma_values = [
        0,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        1,
    ]
    for sigma in sigma_values:
        print(f"{sigma=}")
        (
            _flaky_train,
            _non_flaky_train,
            _flaky_test,
            _non_flaky_test,
            avg_p,
            avg_r,
            storage,
            avg_t_prep,
            avg_t_pred,
        ) = flast_knn(out_dir, project_base_path, project_name, kf, dim, eps, k, sigma, params)
        with (out_dir / out_file).open("a") as fo:
            fo.write(
                f"{params['metric']},{k},{sigma},{eps},{avg_p},{avg_r},"
                f"{storage},{avg_t_prep},{avg_t_pred}\n"
            )

    # TRAINING SET SIZE
    out_file = "params-training.csv"
    with (out_dir / out_file).open("w") as fo:
        fo.write(
            "trainingSetSize,k,sigma,eps,precision,recall,storage,preparationTime,predictionTime\n"
        )

    k = 7
    dim = 0
    eps = 0.3
    params = {"algorithm": "brute", "metric": "cosine", "weights": "uniform"}
    num_split = 30

    for test_set_size in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        training_set_size = 1 - test_set_size
        for sigma in [0.5, 0.95]:
            kf = StratifiedShuffleSplit(n_splits=num_split, test_size=test_set_size)
            print(f"{k=}, {sigma=}, {test_set_size=}")
            (
                _flaky_train,
                _non_flaky_train,
                _flaky_test,
                _non_flaky_test,
                avg_p,
                avg_r,
                storage,
                avg_t_prep,
                avg_t_pred,
            ) = flast_knn(out_dir, project_base_path, project_name, kf, dim, eps, k, sigma, params)
            with (out_dir / out_file).open("a") as fo:
                fo.write(
                    f"{training_set_size},{k},{sigma},{eps},{avg_p},{avg_r},"
                    f"{storage},{avg_t_prep},{avg_t_pred}\n"
                )
