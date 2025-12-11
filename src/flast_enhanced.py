"""FLAST Enhanced: Improved Static Prediction of Test Flakiness.

This module extends the original FLAST algorithm with:
1. Class imbalance handling (SMOTE, class weights, adaptive thresholds)
2. Enhanced feature representation (TF-IDF, n-grams, structural features)
3. Flakiness-specific feature extraction (concurrency, timing, I/O patterns)
4. Improved edge case handling with class prior probabilities
5. Hybrid features support (metadata, historical data)
"""

from __future__ import annotations

import re
import time
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.sparse import hstack, issparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import (
    SparseRandomProjection,
    johnson_lindenstrauss_min_dim,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray
    from scipy.sparse import spmatrix


###############################################################################
# Flakiness Pattern Definitions
###############################################################################

# Patterns that indicate potential test flakiness causes
FLAKINESS_PATTERNS: dict[str, list[str]] = {
    # Concurrency-related patterns
    "concurrency": [
        r"\bsynchronized\b",
        r"\bThread\b",
        r"\bRunnable\b",
        r"\bExecutorService\b",
        r"\bExecutors\b",
        r"\bFuture\b",
        r"\bCallable\b",
        r"\bCountDownLatch\b",
        r"\bSemaphore\b",
        r"\bCyclicBarrier\b",
        r"\bReentrantLock\b",
        r"\bAtomicInteger\b",
        r"\bAtomicBoolean\b",
        r"\bAtomicReference\b",
        r"\bvolatile\b",
        r"\bwait\s*\(",
        r"\bnotify\s*\(",
        r"\bnotifyAll\s*\(",
        r"\bjoin\s*\(",
        r"\bThreadPool\b",
        r"\bparallel\b",
        r"\bConcurrent\w+\b",
    ],
    # Timing-related patterns
    "timing": [
        r"\bThread\.sleep\b",
        r"\bsleep\s*\(",
        r"\btimeout\b",
        r"\bTimeout\b",
        r"\bSystem\.currentTimeMillis\b",
        r"\bSystem\.nanoTime\b",
        r"\bDuration\b",
        r"\bInstant\b",
        r"\bLocalDateTime\b",
        r"\bZonedDateTime\b",
        r"\bDate\b",
        r"\bCalendar\b",
        r"\bTimeUnit\b",
        r"\bScheduled\w*\b",
        r"\bTimer\b",
        r"\bTimerTask\b",
        r"\bawait\b",
        r"\bawaitTermination\b",
    ],
    # Randomness-related patterns
    "randomness": [
        r"\bRandom\b",
        r"\bMath\.random\b",
        r"\bUUID\b",
        r"\bSecureRandom\b",
        r"\bThreadLocalRandom\b",
        r"\bshuffle\b",
        r"\bnextInt\b",
        r"\bnextDouble\b",
        r"\bnextBoolean\b",
        r"\bnextLong\b",
    ],
    # I/O and Network patterns
    "io_network": [
        r"\bFile\b",
        r"\bFiles\b",
        r"\bPath\b",
        r"\bPaths\b",
        r"\bInputStream\b",
        r"\bOutputStream\b",
        r"\bReader\b",
        r"\bWriter\b",
        r"\bBuffered\w+\b",
        r"\bFileChannel\b",
        r"\bSocket\b",
        r"\bServerSocket\b",
        r"\bURL\b",
        r"\bURI\b",
        r"\bHttpClient\b",
        r"\bHttpURLConnection\b",
        r"\bRestTemplate\b",
        r"\bWebClient\b",
        r"\bOkHttpClient\b",
        r"\bRetrofit\b",
        r"\bConnection\b",
        r"\bDataSource\b",
        r"\bResultSet\b",
        r"\bStatement\b",
        r"\bPreparedStatement\b",
        r"\bDriverManager\b",
        r"\bJDBC\b",
    ],
    # External dependencies and mocking
    "external_deps": [
        r"@Mock\b",
        r"@Spy\b",
        r"@InjectMocks\b",
        r"\bMockito\b",
        r"\bwhen\s*\(",
        r"\bverify\s*\(",
        r"\bdoReturn\b",
        r"\bdoThrow\b",
        r"\bmock\s*\(",
        r"\bEasyMock\b",
        r"\bPowerMock\b",
        r"\bWireMock\b",
        r"\bMockServer\b",
        r"\bTestContainers\b",
        r"@Autowired\b",
        r"@Inject\b",
        r"@Resource\b",
    ],
    # Test order and shared state
    "test_order": [
        r"@Before\b",
        r"@After\b",
        r"@BeforeEach\b",
        r"@AfterEach\b",
        r"@BeforeAll\b",
        r"@AfterAll\b",
        r"@BeforeClass\b",
        r"@AfterClass\b",
        r"\bstatic\s+\w+\s+\w+\s*=",
        r"\bstatic\s+final\b",
        r"\bsingleton\b",
        r"\bSingleton\b",
        r"@Shared\b",
    ],
    # Async operations
    "async_ops": [
        r"\bCompletableFuture\b",
        r"\bCompletionStage\b",
        r"\bForkJoinPool\b",
        r"\bForkJoinTask\b",
        r"\basync\b",
        r"\bCallback\b",
        r"\bListener\b",
        r"\bObserver\b",
        r"\bObservable\b",
        r"\bFlowable\b",
        r"\bMono\b",
        r"\bFlux\b",
        r"\bPublisher\b",
        r"\bSubscriber\b",
        r"\bReactive\b",
    ],
    # Assertions
    "assertions": [
        r"\bassertEquals\b",
        r"\bassertTrue\b",
        r"\bassertFalse\b",
        r"\bassertNull\b",
        r"\bassertNotNull\b",
        r"\bassertThat\b",
        r"\bassertThrows\b",
        r"\bexpectedException\b",
        r"\bexpect\s*\(",
        r"\bverify\s*\(",
        r"\bassertArrayEquals\b",
        r"\bassertSame\b",
        r"\bassertNotSame\b",
    ],
}


###############################################################################
# Configuration
###############################################################################


@dataclass
class EnhancedConfig:
    """Configuration for Enhanced FLAST.

    Attributes:
        use_tfidf: Use TF-IDF instead of raw counts.
        ngram_range: Range for n-grams (min_n, max_n).
        use_flakiness_features: Extract flakiness-specific features.
        use_structural_features: Extract structural code features.
        handle_imbalance: Method to handle class imbalance ('smote', 'weights', 'none').
        use_prior_in_edge_cases: Use class prior for edge case predictions.
        dim: Target dimensionality for projection (0 = auto).
        eps: Error tolerance for Johnson-Lindenstrauss projection.
        k: Number of neighbors.
        sigma: Decision threshold.
        algorithm: k-NN algorithm ('brute', 'ball_tree', 'kd_tree').
        metric: Distance metric ('cosine', 'euclidean').
        weights: Neighbor weighting ('distance', 'uniform').
        smote_k_neighbors: Number of neighbors for SMOTE.
        class_weight_ratio: Weight ratio for flaky class (higher = more important).
    """

    use_tfidf: bool = True
    ngram_range: tuple[int, int] = (1, 2)
    use_flakiness_features: bool = True
    use_structural_features: bool = True
    handle_imbalance: str = "weights"  # 'smote', 'weights', 'none'
    use_prior_in_edge_cases: bool = True
    dim: int = 0
    eps: float = 0.3
    k: int = 7
    sigma: float = 0.5
    algorithm: str = "brute"
    metric: str = "cosine"
    weights: str = "distance"
    smote_k_neighbors: int = 5
    class_weight_ratio: float = 1.0  # Auto-computed if 0


@dataclass
class HybridFeatures:
    """Hybrid features from external sources (historical data, metadata).

    Attributes:
        historical_flakiness_rate: Past flakiness rate for each test (0-1).
        code_change_frequency: Recent modification frequency (normalized).
        execution_time_variance: Variance in test execution time (normalized).
        dependency_flakiness_score: Flakiness score of dependent tests.
        test_age_days: Age of the test in days (normalized).
        file_change_count: Number of recent file changes.
        author_flakiness_rate: Historical flakiness rate of the test author.
    """

    historical_flakiness_rate: list[float] = field(default_factory=list)
    code_change_frequency: list[float] = field(default_factory=list)
    execution_time_variance: list[float] = field(default_factory=list)
    dependency_flakiness_score: list[float] = field(default_factory=list)
    test_age_days: list[float] = field(default_factory=list)
    file_change_count: list[float] = field(default_factory=list)
    author_flakiness_rate: list[float] = field(default_factory=list)


###############################################################################
# Data Loading
###############################################################################


def get_data_points(path: Path) -> list[str]:
    """Read tokenized test methods from a directory.

    Args:
        path: Directory containing tokenized test method files.

    Returns:
        List of tokenized test methods as strings.
    """
    data_points: list[str] = []
    for data_point_file in sorted(path.iterdir()):
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


###############################################################################
# Feature Extraction
###############################################################################


def extract_flakiness_features(data_points: list[str]) -> NDArray[np.float64]:
    """Extract flakiness-specific features from test code.

    Counts occurrences of patterns associated with test flakiness.

    Args:
        data_points: List of test method source code.

    Returns:
        Array of shape (n_samples, n_features) with pattern counts.
    """
    n_samples = len(data_points)
    n_categories = len(FLAKINESS_PATTERNS)

    features = np.zeros((n_samples, n_categories), dtype=np.float64)

    for i, code in enumerate(data_points):
        for j, (_category, patterns) in enumerate(FLAKINESS_PATTERNS.items()):
            count = 0
            for pattern in patterns:
                count += len(re.findall(pattern, code, re.IGNORECASE))
            features[i, j] = count

    return features


def extract_structural_features(data_points: list[str]) -> NDArray[np.float64]:
    """Extract structural features from test code.

    Extracts features like:
    - Code length (characters, lines, tokens)
    - Method complexity indicators
    - Nesting depth estimates

    Args:
        data_points: List of test method source code.

    Returns:
        Array of shape (n_samples, n_features) with structural features.
    """
    n_samples = len(data_points)
    # Features: char_count, line_count, token_count, avg_line_length,
    #           brace_depth_max, paren_count, semicolon_count, dot_count
    n_features = 8

    features = np.zeros((n_samples, n_features), dtype=np.float64)

    for i, code in enumerate(data_points):
        # Character count
        features[i, 0] = len(code)

        # Line count
        lines = code.split("\n")
        features[i, 1] = len(lines)

        # Token count (simple whitespace split)
        tokens = code.split()
        features[i, 2] = len(tokens)

        # Average line length
        features[i, 3] = features[i, 0] / max(features[i, 1], 1)

        # Maximum brace nesting depth
        max_depth = 0
        current_depth = 0
        for char in code:
            if char == "{":
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == "}":
                current_depth = max(0, current_depth - 1)
        features[i, 4] = max_depth

        # Parenthesis count (indicates method calls)
        features[i, 5] = code.count("(")

        # Semicolon count (indicates statements)
        features[i, 6] = code.count(";")

        # Dot count (indicates method chaining/field access)
        features[i, 7] = code.count(".")

    return features


def extract_hybrid_features(
    hybrid_data: HybridFeatures | None, n_samples: int
) -> NDArray[np.float64] | None:
    """Extract hybrid features from external metadata.

    Args:
        hybrid_data: HybridFeatures object with external data.
        n_samples: Expected number of samples.

    Returns:
        Array of shape (n_samples, n_features) or None if no data.
    """
    if hybrid_data is None:
        return None

    feature_lists = [
        hybrid_data.historical_flakiness_rate,
        hybrid_data.code_change_frequency,
        hybrid_data.execution_time_variance,
        hybrid_data.dependency_flakiness_score,
        hybrid_data.test_age_days,
        hybrid_data.file_change_count,
        hybrid_data.author_flakiness_rate,
    ]

    # Check which features are available
    valid_features = []
    for feat_list in feature_lists:
        if len(feat_list) == n_samples:
            valid_features.append(np.array(feat_list, dtype=np.float64))

    if not valid_features:
        return None

    return np.column_stack(valid_features)


###############################################################################
# Vectorization
###############################################################################


def enhanced_vectorization(
    data_points: list[str],
    config: EnhancedConfig,
    hybrid_features: HybridFeatures | None = None,
) -> tuple[NDArray[Any] | spmatrix, dict[str, Any]]:
    """Enhanced vectorization with TF-IDF, n-grams, and additional features.

    Args:
        data_points: List of tokenized test methods.
        config: Configuration for vectorization.
        hybrid_features: Optional hybrid features from external sources.

    Returns:
        Tuple of (feature_matrix, metadata_dict).
        metadata_dict contains the vectorizer and scaler for later use.
    """
    metadata: dict[str, Any] = {}

    # Text vectorization
    if config.use_tfidf:
        vectorizer = TfidfVectorizer(ngram_range=config.ngram_range)
    else:
        vectorizer = CountVectorizer(ngram_range=config.ngram_range)

    text_features = vectorizer.fit_transform(data_points)
    metadata["vectorizer"] = vectorizer

    feature_parts = [text_features]

    # Flakiness-specific features
    if config.use_flakiness_features:
        flaky_features = extract_flakiness_features(data_points)
        # Normalize
        scaler_flaky = StandardScaler()
        flaky_features_scaled = scaler_flaky.fit_transform(flaky_features)
        metadata["scaler_flaky"] = scaler_flaky
        feature_parts.append(flaky_features_scaled)

    # Structural features
    if config.use_structural_features:
        struct_features = extract_structural_features(data_points)
        # Normalize
        scaler_struct = StandardScaler()
        struct_features_scaled = scaler_struct.fit_transform(struct_features)
        metadata["scaler_struct"] = scaler_struct
        feature_parts.append(struct_features_scaled)

    # Hybrid features
    hybrid_array = extract_hybrid_features(hybrid_features, len(data_points))
    if hybrid_array is not None:
        scaler_hybrid = StandardScaler()
        hybrid_scaled = scaler_hybrid.fit_transform(hybrid_array)
        metadata["scaler_hybrid"] = scaler_hybrid
        metadata["has_hybrid"] = True
        feature_parts.append(hybrid_scaled)
    else:
        metadata["has_hybrid"] = False

    # Combine all features
    combined = feature_parts[0] if len(feature_parts) == 1 else hstack(feature_parts)

    # Dimensionality reduction
    if config.eps > 0:
        if config.dim <= 0:
            dim = johnson_lindenstrauss_min_dim(combined.shape[0], eps=config.eps)
        else:
            dim = config.dim

        # Ensure dim doesn't exceed feature count
        max_dim = combined.shape[1]
        dim = min(dim, max_dim)

        srp = SparseRandomProjection(n_components=dim)
        combined = srp.fit_transform(combined)
        metadata["srp"] = srp

    return combined, metadata


def transform_test_data(
    data_points: list[str],
    config: EnhancedConfig,
    metadata: dict[str, Any],
    hybrid_features: HybridFeatures | None = None,
) -> NDArray[Any] | spmatrix:
    """Transform test data using fitted transformers.

    Args:
        data_points: List of tokenized test methods.
        config: Configuration for vectorization.
        metadata: Metadata from training (contains fitted transformers).
        hybrid_features: Optional hybrid features from external sources.

    Returns:
        Transformed feature matrix.
    """
    # Text vectorization using fitted vectorizer
    vectorizer = metadata["vectorizer"]
    text_features = vectorizer.transform(data_points)

    feature_parts = [text_features]

    # Flakiness-specific features
    if config.use_flakiness_features and "scaler_flaky" in metadata:
        flaky_features = extract_flakiness_features(data_points)
        scaler_flaky = metadata["scaler_flaky"]
        flaky_features_scaled = scaler_flaky.transform(flaky_features)
        feature_parts.append(flaky_features_scaled)

    # Structural features
    if config.use_structural_features and "scaler_struct" in metadata:
        struct_features = extract_structural_features(data_points)
        scaler_struct = metadata["scaler_struct"]
        struct_features_scaled = scaler_struct.transform(struct_features)
        feature_parts.append(struct_features_scaled)

    # Hybrid features (only if training had hybrid features)
    if metadata.get("has_hybrid", False) and "scaler_hybrid" in metadata:
        hybrid_array = extract_hybrid_features(hybrid_features, len(data_points))
        if hybrid_array is not None:
            scaler_hybrid = metadata["scaler_hybrid"]
            hybrid_scaled = scaler_hybrid.transform(hybrid_array)
            feature_parts.append(hybrid_scaled)
        else:
            # Fill with zeros if hybrid features were in training but not in test
            n_hybrid_features = metadata["scaler_hybrid"].n_features_in_
            feature_parts.append(np.zeros((len(data_points), n_hybrid_features)))

    # Combine all features
    combined = feature_parts[0] if len(feature_parts) == 1 else hstack(feature_parts)

    # Apply dimensionality reduction if it was used during training
    if "srp" in metadata:
        combined = metadata["srp"].transform(combined)

    return combined


###############################################################################
# Class Imbalance Handling
###############################################################################


def apply_smote(
    train_data: NDArray[Any],
    train_labels: NDArray[np.int64],
    k_neighbors: int = 5,
) -> tuple[NDArray[Any], NDArray[np.int64]]:
    """Apply SMOTE to balance training data.

    Uses a simple implementation that doesn't require imblearn.

    Args:
        train_data: Training feature matrix.
        train_labels: Training labels.
        k_neighbors: Number of neighbors for SMOTE.

    Returns:
        Tuple of (balanced_data, balanced_labels).
    """
    # Count classes
    n_minority = np.sum(train_labels == 1)
    n_majority = np.sum(train_labels == 0)

    if n_minority == 0 or n_minority >= n_majority:
        return train_data, train_labels

    # Get minority class samples
    minority_indices = np.where(train_labels == 1)[0]
    minority_samples = train_data[minority_indices]

    # Number of synthetic samples to generate
    n_synthetic = n_majority - n_minority

    # Generate synthetic samples
    synthetic_samples = []
    rng = np.random.default_rng(42)

    for _ in range(n_synthetic):
        # Pick a random minority sample
        idx = rng.integers(0, len(minority_samples))
        sample = minority_samples[idx]

        # Find k nearest neighbors within minority class
        distances = np.linalg.norm(minority_samples - sample, axis=1)
        neighbor_indices = np.argsort(distances)[1 : k_neighbors + 1]

        if len(neighbor_indices) == 0:
            continue

        # Pick a random neighbor
        neighbor_idx = rng.choice(neighbor_indices)
        neighbor = minority_samples[neighbor_idx]

        # Generate synthetic sample along the line
        alpha = rng.random()
        synthetic = sample + alpha * (neighbor - sample)
        synthetic_samples.append(synthetic)

    if synthetic_samples:
        synthetic_array = np.array(synthetic_samples)
        balanced_data = np.vstack([train_data, synthetic_array])
        balanced_labels = np.concatenate(
            [train_labels, np.ones(len(synthetic_samples), dtype=np.int64)]
        )
    else:
        balanced_data = train_data
        balanced_labels = train_labels

    return balanced_data, balanced_labels


def compute_class_weights(
    train_labels: list[int] | NDArray[np.int64], ratio: float = 0.0
) -> dict[int, float]:
    """Compute class weights for imbalanced data.

    Args:
        train_labels: Training labels.
        ratio: Fixed weight ratio (0 = auto-compute based on class frequency).

    Returns:
        Dictionary mapping class labels to weights.
    """
    labels_array = np.array(train_labels)
    n_samples = len(labels_array)
    n_flaky = np.sum(labels_array == 1)
    n_non_flaky = n_samples - n_flaky

    if ratio > 0:
        return {0: 1.0, 1: ratio}

    if n_flaky == 0 or n_non_flaky == 0:
        return {0: 1.0, 1: 1.0}

    # Balanced class weights
    weight_flaky = n_samples / (2 * n_flaky)
    weight_non_flaky = n_samples / (2 * n_non_flaky)

    return {0: weight_non_flaky, 1: weight_flaky}


###############################################################################
# Classification
###############################################################################


def enhanced_classification(
    train_data: NDArray[Any] | spmatrix,
    train_labels: list[int] | NDArray[np.int64],
    test_data: NDArray[Any] | spmatrix,
    config: EnhancedConfig,
    class_prior: float | None = None,
) -> tuple[float, float, list[int], list[float]]:
    """Enhanced k-NN classification with improved features.

    Args:
        train_data: Vectorized training data.
        train_labels: Training labels (1 for flaky, 0 for non-flaky).
        test_data: Vectorized test data.
        config: Configuration for classification.
        class_prior: Prior probability of flaky class (computed if None).

    Returns:
        Tuple of (train_time, test_time, predicted_labels, confidence_scores).
    """
    train_labels_array = np.array(train_labels, dtype=np.int64)

    # Convert sparse to dense if needed for SMOTE
    train_data_dense = train_data.toarray() if issparse(train_data) else np.asarray(train_data)

    test_data_dense = test_data.toarray() if issparse(test_data) else np.asarray(test_data)

    # Compute class prior for edge cases
    if class_prior is None:
        n_flaky = np.sum(train_labels_array == 1)
        class_prior = n_flaky / len(train_labels_array) if len(train_labels_array) > 0 else 0.5

    # Apply SMOTE if configured
    if config.handle_imbalance == "smote":
        train_data_balanced, train_labels_balanced = apply_smote(
            train_data_dense, train_labels_array, config.smote_k_neighbors
        )
    else:
        train_data_balanced = train_data_dense
        train_labels_balanced = train_labels_array

    # Training
    t0 = time.perf_counter()
    knn = KNeighborsClassifier(
        algorithm=config.algorithm,
        metric=config.metric,
        weights=config.weights,
        n_neighbors=min(config.k, len(train_labels_balanced)),
        n_jobs=1,
    )
    knn.fit(train_data_balanced, train_labels_balanced)
    t1 = time.perf_counter()
    train_time = t1 - t0

    # Compute class weights if needed
    class_weights = None
    if config.handle_imbalance == "weights":
        class_weights = compute_class_weights(train_labels_balanced, config.class_weight_ratio)

    # Prediction with enhanced voting
    t0 = time.perf_counter()
    predict_labels: list[int] = []
    confidence_scores: list[float] = []
    neighbor_dist, neighbor_ind = knn.kneighbors(test_data_dense)

    use_distance_weights = config.weights == "distance"

    for distances, indices in zip(neighbor_dist, neighbor_ind, strict=True):
        phi, psi = 0.0, 0.0

        for distance, neighbor in zip(distances, indices, strict=True):
            if use_distance_weights:
                d_inv = (1 / distance) if distance != 0 else float("inf")
            else:
                d_inv = 1.0

            # Apply class weights if configured
            if class_weights is not None:
                label = train_labels_balanced[neighbor]
                d_inv *= class_weights[label]

            if train_labels_balanced[neighbor] == 1:
                phi += d_inv
            else:
                psi += d_inv

        # Compute prediction with improved edge case handling
        prediction, confidence = _compute_prediction_enhanced(
            phi, psi, config.sigma, class_prior, config.use_prior_in_edge_cases
        )
        predict_labels.append(prediction)
        confidence_scores.append(confidence)

    t1 = time.perf_counter()
    test_time = t1 - t0

    return train_time, test_time, predict_labels, confidence_scores


def _compute_prediction_enhanced(
    phi: float,
    psi: float,
    sigma: float,
    class_prior: float,
    use_prior: bool,
) -> tuple[int, float]:
    """Compute prediction with improved edge case handling.

    Args:
        phi: Weighted sum of flaky neighbor votes.
        psi: Weighted sum of non-flaky neighbor votes.
        sigma: Decision threshold.
        class_prior: Prior probability of flaky class.
        use_prior: Whether to use class prior for edge cases.

    Returns:
        Tuple of (prediction, confidence).
        prediction: 1 if predicted flaky, 0 otherwise.
        confidence: Confidence score (0-1).
    """
    inf = float("inf")

    # Edge case: both infinite (exact matches in both classes)
    if phi == inf and psi == inf:
        # Use class prior to break tie
        if use_prior:
            prediction = 1 if class_prior >= sigma else 0
            confidence = class_prior
        else:
            prediction = 0
            confidence = 0.5
        return prediction, confidence

    # Edge case: only non-flaky neighbors at distance 0
    if psi == inf:
        return 0, 1.0

    # Edge case: only flaky neighbors at distance 0
    if phi == inf:
        return 1, 1.0

    # Edge case: no neighbors (shouldn't happen but handle gracefully)
    if (phi + psi) == 0:
        if use_prior:
            prediction = 1 if class_prior >= sigma else 0
            confidence = class_prior
        else:
            prediction = 0
            confidence = 0.5
        return prediction, confidence

    # Normal case: compute ratio
    ratio = phi / (phi + psi)
    confidence = ratio if ratio >= 0.5 else 1 - ratio

    if ratio >= sigma:
        return 1, confidence
    return 0, confidence


###############################################################################
# Metrics Computation
###############################################################################


def compute_results(
    test_labels: list[int], predict_labels: list[int]
) -> tuple[float | str, float | str]:
    """Compute precision and recall for predictions.

    Args:
        test_labels: Ground truth labels.
        predict_labels: Predicted labels.

    Returns:
        Tuple of (precision, recall).
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


def compute_results_extended(
    test_labels: list[int], predict_labels: list[int]
) -> dict[str, float | str]:
    """Compute extended metrics for predictions.

    Args:
        test_labels: Ground truth labels.
        predict_labels: Predicted labels.

    Returns:
        Dictionary with precision, recall, f1, and support.
    """
    results: dict[str, float | str] = {}

    with warnings.catch_warnings():
        warnings.filterwarnings("error")

        try:
            results["precision"] = float(precision_score(test_labels, predict_labels))
        except (ValueError, Warning):
            results["precision"] = "-"

        try:
            results["recall"] = float(recall_score(test_labels, predict_labels))
        except (ValueError, Warning):
            results["recall"] = "-"

        try:
            results["f1"] = float(f1_score(test_labels, predict_labels))
        except (ValueError, Warning):
            results["f1"] = "-"

    results["n_flaky"] = sum(test_labels)
    results["n_non_flaky"] = len(test_labels) - sum(test_labels)
    results["n_predicted_flaky"] = sum(predict_labels)

    return results


###############################################################################
# High-Level API
###############################################################################


class EnhancedFLAST:
    """Enhanced FLAST classifier with all improvements.

    This class provides a scikit-learn compatible interface for the
    enhanced FLAST algorithm.

    Example:
        >>> config = EnhancedConfig(use_tfidf=True, handle_imbalance='smote')
        >>> clf = EnhancedFLAST(config)
        >>> clf.fit(train_texts, train_labels)
        >>> predictions = clf.predict(test_texts)
        >>> precision, recall = clf.score(test_texts, test_labels)
    """

    def __init__(self, config: EnhancedConfig | None = None):
        """Initialize Enhanced FLAST.

        Args:
            config: Configuration object. Uses defaults if None.
        """
        self.config = config or EnhancedConfig()
        self._metadata: dict[str, Any] = {}
        self._train_data: NDArray[Any] | None = None
        self._train_labels: NDArray[np.int64] | None = None
        self._class_prior: float = 0.5
        self._is_fitted: bool = False

    def fit(
        self,
        data_points: list[str],
        labels: list[int],
        hybrid_features: HybridFeatures | None = None,
    ) -> EnhancedFLAST:
        """Fit the classifier on training data.

        Args:
            data_points: List of tokenized test methods.
            labels: Labels (1 for flaky, 0 for non-flaky).
            hybrid_features: Optional hybrid features.

        Returns:
            Self for method chaining.
        """
        # Vectorize
        vectors, self._metadata = enhanced_vectorization(
            data_points, self.config, hybrid_features
        )

        # Convert to array
        if issparse(vectors):
            self._train_data = vectors.toarray()
        else:
            self._train_data = np.asarray(vectors)

        self._train_labels = np.array(labels, dtype=np.int64)

        # Compute class prior
        n_flaky = np.sum(self._train_labels == 1)
        n_total = len(self._train_labels)
        self._class_prior = n_flaky / n_total if n_total > 0 else 0.5

        self._is_fitted = True
        return self

    def predict(
        self,
        data_points: list[str],
        hybrid_features: HybridFeatures | None = None,
    ) -> list[int]:
        """Predict labels for test data.

        Args:
            data_points: List of tokenized test methods.
            hybrid_features: Optional hybrid features.

        Returns:
            List of predicted labels.
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier must be fitted before prediction")

        # Transform using fitted transformers
        vectors = transform_test_data(
            data_points, self.config, self._metadata, hybrid_features
        )

        test_data = vectors.toarray() if issparse(vectors) else np.asarray(vectors)

        _, _, predictions, _ = enhanced_classification(
            self._train_data,
            self._train_labels.tolist(),
            test_data,
            self.config,
            self._class_prior,
        )

        return predictions

    def predict_proba(
        self,
        data_points: list[str],
        hybrid_features: HybridFeatures | None = None,
    ) -> list[float]:
        """Predict confidence scores for test data.

        Args:
            data_points: List of tokenized test methods.
            hybrid_features: Optional hybrid features.

        Returns:
            List of confidence scores.
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier must be fitted before prediction")

        # Transform using fitted transformers
        vectors = transform_test_data(
            data_points, self.config, self._metadata, hybrid_features
        )

        test_data = vectors.toarray() if issparse(vectors) else np.asarray(vectors)

        _, _, _, confidence = enhanced_classification(
            self._train_data,
            self._train_labels.tolist(),
            test_data,
            self.config,
            self._class_prior,
        )

        return confidence

    def score(
        self,
        data_points: list[str],
        labels: list[int],
        hybrid_features: HybridFeatures | None = None,
    ) -> tuple[float | str, float | str]:
        """Compute precision and recall on test data.

        Args:
            data_points: List of tokenized test methods.
            labels: Ground truth labels.
            hybrid_features: Optional hybrid features.

        Returns:
            Tuple of (precision, recall).
        """
        predictions = self.predict(data_points, hybrid_features)
        return compute_results(labels, predictions)

    def score_extended(
        self,
        data_points: list[str],
        labels: list[int],
        hybrid_features: HybridFeatures | None = None,
    ) -> dict[str, float | str]:
        """Compute extended metrics on test data.

        Args:
            data_points: List of tokenized test methods.
            labels: Ground truth labels.
            hybrid_features: Optional hybrid features.

        Returns:
            Dictionary with precision, recall, f1, and support.
        """
        predictions = self.predict(data_points, hybrid_features)
        return compute_results_extended(labels, predictions)


###############################################################################
# Utility Functions
###############################################################################


def get_flakiness_pattern_names() -> list[str]:
    """Get list of flakiness pattern category names.

    Returns:
        List of category names.
    """
    return list(FLAKINESS_PATTERNS.keys())


def analyze_flakiness_patterns(code: str) -> dict[str, int]:
    """Analyze a single code sample for flakiness patterns.

    Args:
        code: Test method source code.

    Returns:
        Dictionary mapping pattern categories to counts.
    """
    result = {}
    for category, patterns in FLAKINESS_PATTERNS.items():
        count = 0
        for pattern in patterns:
            count += len(re.findall(pattern, code, re.IGNORECASE))
        result[category] = count
    return result
