"""FLAST: Fast Static Prediction of Test Flakiness.

This package implements the FLAST algorithm for predicting test flakiness
using k-Nearest Neighbors classification.
"""

from src.flast import (
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
    "computeResults",
    "compute_results",
    "flastClassification",
    "flastVectorization",
    "flast_classification",
    "flast_vectorization",
    "getDataPoints",
    "getDataPointsInfo",
    "get_data_points",
    "get_data_points_info",
]

__version__ = "1.0.0"
