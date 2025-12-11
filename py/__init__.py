"""FLAST: Fast Static Prediction of Test Flakiness.

This package implements the FLAST algorithm for predicting test flakiness
using k-Nearest Neighbors classification.
"""

from py.flast import (
    compute_results,
    computeResults,
    flast_classification,
    flast_vectorization,
    flastClassification,
    flastVectorization,
    get_data_points,
    get_data_points_info,
    getDataPoints,
    getDataPointsInfo,
)

__all__ = [
    # Modern API
    "get_data_points",
    "get_data_points_info",
    "compute_results",
    "flast_vectorization",
    "flast_classification",
    # Legacy API (deprecated)
    "getDataPoints",
    "getDataPointsInfo",
    "computeResults",
    "flastVectorization",
    "flastClassification",
]

__version__ = "1.0.0"
