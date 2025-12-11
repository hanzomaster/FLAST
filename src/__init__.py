"""FLAST: Fast Static Prediction of Test Flakiness.

This package implements the FLAST algorithm for predicting test flakiness
using k-Nearest Neighbors classification.

Includes both the original FLAST implementation and an enhanced version with:
- Class imbalance handling (SMOTE, class weights)
- Enhanced feature representation (TF-IDF, n-grams)
- Flakiness-specific feature extraction
- Improved edge case handling with class priors
- Hybrid features support (metadata, historical data)
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
from src.flast_enhanced import (
    FLAKINESS_PATTERNS,
    EnhancedConfig,
    EnhancedFLAST,
    HybridFeatures,
    analyze_flakiness_patterns,
    apply_smote,
    compute_class_weights,
    compute_results_extended,
    enhanced_classification,
    enhanced_vectorization,
    extract_flakiness_features,
    extract_structural_features,
    get_flakiness_pattern_names,
    transform_test_data,
)

__all__ = [
    "FLAKINESS_PATTERNS",
    # Enhanced FLAST
    "EnhancedConfig",
    "EnhancedFLAST",
    "HybridFeatures",
    "analyze_flakiness_patterns",
    "apply_smote",
    # Original FLAST
    "computeResults",
    "compute_class_weights",
    "compute_results",
    "compute_results_extended",
    "enhanced_classification",
    "enhanced_vectorization",
    "extract_flakiness_features",
    "extract_structural_features",
    "flastClassification",
    "flastVectorization",
    "flast_classification",
    "flast_vectorization",
    "getDataPoints",
    "getDataPointsInfo",
    "get_data_points",
    "get_data_points_info",
    "get_flakiness_pattern_names",
    "transform_test_data",
]

__version__ = "2.0.0"
