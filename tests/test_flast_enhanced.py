"""Unit tests for the Enhanced FLAST algorithm."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import flast_enhanced
from flast_enhanced import (
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
)


class TestFlakinessPatterns:
    """Tests for flakiness pattern detection."""

    def test_flakiness_patterns_defined(self) -> None:
        """Test that flakiness patterns are defined."""
        assert len(FLAKINESS_PATTERNS) > 0
        assert "concurrency" in FLAKINESS_PATTERNS
        assert "timing" in FLAKINESS_PATTERNS
        assert "randomness" in FLAKINESS_PATTERNS
        assert "io_network" in FLAKINESS_PATTERNS

    def test_get_flakiness_pattern_names(self) -> None:
        """Test getting pattern category names."""
        names = get_flakiness_pattern_names()
        assert isinstance(names, list)
        assert len(names) == len(FLAKINESS_PATTERNS)
        assert "concurrency" in names

    def test_analyze_flakiness_patterns_concurrency(self) -> None:
        """Test detecting concurrency patterns."""
        code = """
        public void testConcurrent() {
            Thread thread = new Thread();
            synchronized (lock) {
                thread.start();
            }
            thread.join();
        }
        """
        result = analyze_flakiness_patterns(code)

        assert result["concurrency"] > 0
        assert "Thread" in code or result["concurrency"] >= 2

    def test_analyze_flakiness_patterns_timing(self) -> None:
        """Test detecting timing patterns."""
        code = """
        public void testTiming() {
            Thread.sleep(1000);
            long start = System.currentTimeMillis();
            await(timeout);
        }
        """
        result = analyze_flakiness_patterns(code)

        assert result["timing"] > 0

    def test_analyze_flakiness_patterns_randomness(self) -> None:
        """Test detecting randomness patterns."""
        code = """
        public void testRandom() {
            Random rand = new Random();
            int value = rand.nextInt(100);
            UUID uuid = UUID.randomUUID();
        }
        """
        result = analyze_flakiness_patterns(code)

        assert result["randomness"] > 0

    def test_analyze_flakiness_patterns_io_network(self) -> None:
        """Test detecting I/O and network patterns."""
        code = """
        public void testNetwork() {
            URL url = new URL("http://example.com");
            HttpClient client = HttpClient.newBuilder().build();
            File file = new File("test.txt");
        }
        """
        result = analyze_flakiness_patterns(code)

        assert result["io_network"] > 0

    def test_analyze_flakiness_patterns_clean_code(self) -> None:
        """Test that clean code has low pattern counts."""
        code = """
        public void testSimple() {
            int a = 1;
            int b = 2;
            assertEquals(3, a + b);
        }
        """
        result = analyze_flakiness_patterns(code)

        # Should have assertion patterns but low flakiness indicators
        assert result["concurrency"] == 0
        assert result["timing"] == 0
        assert result["randomness"] == 0


class TestFeatureExtraction:
    """Tests for feature extraction functions."""

    def test_extract_flakiness_features_shape(self) -> None:
        """Test flakiness feature extraction output shape."""
        data_points = [
            "Thread sleep random",
            "assertEquals assertTrue",
            "File URL Socket",
        ]
        features = extract_flakiness_features(data_points)

        assert features.shape[0] == len(data_points)
        assert features.shape[1] == len(FLAKINESS_PATTERNS)

    def test_extract_flakiness_features_values(self) -> None:
        """Test flakiness feature values are non-negative."""
        data_points = ["Thread.sleep(100)", "Random random = new Random()"]
        features = extract_flakiness_features(data_points)

        assert np.all(features >= 0)

    def test_extract_structural_features_shape(self) -> None:
        """Test structural feature extraction output shape."""
        data_points = [
            "public void test() { int x = 1; }",
            "void method() { if (true) { } }",
        ]
        features = extract_structural_features(data_points)

        assert features.shape[0] == len(data_points)
        assert features.shape[1] == 8  # 8 structural features

    def test_extract_structural_features_values(self) -> None:
        """Test structural feature values."""
        code = "public void test() { int x = 1; x++; }"
        features = extract_structural_features([code])

        # Character count
        assert features[0, 0] == len(code)
        # Line count
        assert features[0, 1] == 1  # Single line
        # Should have some semicolons
        assert features[0, 6] > 0

    def test_extract_structural_features_nesting(self) -> None:
        """Test nesting depth detection."""
        nested_code = "void test() { if (true) { while (x) { } } }"
        flat_code = "void test() { return; }"

        nested_features = extract_structural_features([nested_code])
        flat_features = extract_structural_features([flat_code])

        # Nested code should have higher brace depth
        assert nested_features[0, 4] > flat_features[0, 4]


class TestEnhancedVectorization:
    """Tests for enhanced vectorization."""

    def test_enhanced_vectorization_basic(self) -> None:
        """Test basic enhanced vectorization."""
        data_points = [
            "test method assert equals",
            "thread sleep timeout random",
            "file url socket connection",
        ]
        config = EnhancedConfig(
            use_tfidf=True,
            use_flakiness_features=True,
            use_structural_features=True,
            eps=0,
        )

        vectors, metadata = enhanced_vectorization(data_points, config)

        assert vectors.shape[0] == len(data_points)
        assert vectors.shape[1] > 0
        assert "vectorizer" in metadata

    def test_enhanced_vectorization_tfidf_vs_count(self) -> None:
        """Test TF-IDF vs CountVectorizer."""
        data_points = ["test test test", "test method", "method method"]

        config_tfidf = EnhancedConfig(use_tfidf=True, eps=0)
        config_count = EnhancedConfig(use_tfidf=False, eps=0)

        vectors_tfidf, _ = enhanced_vectorization(data_points, config_tfidf)
        vectors_count, _ = enhanced_vectorization(data_points, config_count)

        # Both should have same number of samples
        assert vectors_tfidf.shape[0] == vectors_count.shape[0]

    def test_enhanced_vectorization_ngrams(self) -> None:
        """Test n-gram feature extraction."""
        data_points = ["thread sleep", "random number"]

        config_unigram = EnhancedConfig(ngram_range=(1, 1), eps=0)
        config_bigram = EnhancedConfig(ngram_range=(1, 2), eps=0)

        vectors_uni, _ = enhanced_vectorization(data_points, config_unigram)
        vectors_bi, _ = enhanced_vectorization(data_points, config_bigram)

        # Bigrams should have more features
        assert vectors_bi.shape[1] >= vectors_uni.shape[1]

    def test_enhanced_vectorization_with_projection(self) -> None:
        """Test vectorization with dimensionality reduction."""
        data_points = ["test"] * 100  # Need enough samples for JL
        config = EnhancedConfig(eps=0.3, dim=10)

        vectors, metadata = enhanced_vectorization(data_points, config)

        assert vectors.shape[0] == len(data_points)
        assert "srp" in metadata  # Sparse random projection was applied


class TestClassImbalanceHandling:
    """Tests for class imbalance handling."""

    def test_apply_smote_basic(self) -> None:
        """Test SMOTE basic functionality."""
        # Imbalanced data: 3 minority, 10 majority
        train_data = np.random.rand(13, 5)
        train_labels = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        _balanced_data, balanced_labels = apply_smote(train_data, train_labels, k_neighbors=2)

        # Should have more samples now
        assert len(balanced_labels) >= len(train_labels)
        # Should have more balanced classes
        n_minority_after = np.sum(balanced_labels == 1)
        assert n_minority_after >= 3

    def test_apply_smote_already_balanced(self) -> None:
        """Test SMOTE with already balanced data."""
        train_data = np.random.rand(10, 5)
        train_labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

        _balanced_data, balanced_labels = apply_smote(train_data, train_labels)

        # Should not change when already balanced
        assert len(balanced_labels) == len(train_labels)

    def test_apply_smote_no_minority(self) -> None:
        """Test SMOTE with no minority class samples."""
        train_data = np.random.rand(10, 5)
        train_labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        _balanced_data, balanced_labels = apply_smote(train_data, train_labels)

        # Should return unchanged
        assert len(balanced_labels) == len(train_labels)

    def test_compute_class_weights_imbalanced(self) -> None:
        """Test class weight computation for imbalanced data."""
        # 2 flaky, 8 non-flaky
        labels = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        weights = compute_class_weights(labels)

        assert weights[1] > weights[0]  # Minority class should have higher weight

    def test_compute_class_weights_balanced(self) -> None:
        """Test class weight computation for balanced data."""
        labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        weights = compute_class_weights(labels)

        assert weights[0] == pytest.approx(weights[1], rel=0.1)

    def test_compute_class_weights_fixed_ratio(self) -> None:
        """Test class weight computation with fixed ratio."""
        labels = [1, 0, 0, 0, 0]
        weights = compute_class_weights(labels, ratio=5.0)

        assert weights[1] == 5.0
        assert weights[0] == 1.0


class TestEnhancedClassification:
    """Tests for enhanced classification."""

    @pytest.fixture
    def sample_enhanced_data(self) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """Sample data for enhanced classification tests."""
        train_data = np.random.rand(20, 10)
        test_data = np.random.rand(5, 10)
        train_labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        return train_data, test_data, train_labels

    def test_enhanced_classification_basic(
        self, sample_enhanced_data: tuple[np.ndarray, np.ndarray, list[int]]
    ) -> None:
        """Test basic enhanced classification."""
        train_data, test_data, train_labels = sample_enhanced_data
        config = EnhancedConfig(handle_imbalance="none")

        train_time, test_time, predictions, confidence = enhanced_classification(
            train_data, train_labels, test_data, config
        )

        assert train_time >= 0
        assert test_time >= 0
        assert len(predictions) == len(test_data)
        assert len(confidence) == len(test_data)
        assert all(p in [0, 1] for p in predictions)
        assert all(0 <= c <= 1 for c in confidence)

    def test_enhanced_classification_with_smote(
        self, sample_enhanced_data: tuple[np.ndarray, np.ndarray, list[int]]
    ) -> None:
        """Test classification with SMOTE."""
        train_data, test_data, train_labels = sample_enhanced_data
        config = EnhancedConfig(handle_imbalance="smote", smote_k_neighbors=3)

        _, _, predictions, _ = enhanced_classification(
            train_data, train_labels, test_data, config
        )

        assert len(predictions) == len(test_data)

    def test_enhanced_classification_with_weights(
        self, sample_enhanced_data: tuple[np.ndarray, np.ndarray, list[int]]
    ) -> None:
        """Test classification with class weights."""
        train_data, test_data, train_labels = sample_enhanced_data
        config = EnhancedConfig(handle_imbalance="weights")

        _, _, predictions, _ = enhanced_classification(
            train_data, train_labels, test_data, config
        )

        assert len(predictions) == len(test_data)

    def test_enhanced_classification_confidence_scores(
        self, sample_enhanced_data: tuple[np.ndarray, np.ndarray, list[int]]
    ) -> None:
        """Test that confidence scores are meaningful."""
        train_data, test_data, train_labels = sample_enhanced_data
        config = EnhancedConfig()

        _, _, _predictions, confidence = enhanced_classification(
            train_data, train_labels, test_data, config
        )

        # Confidence should be between 0 and 1
        assert all(0 <= c <= 1 for c in confidence)


class TestEdgeCaseHandling:
    """Tests for improved edge case handling."""

    def test_compute_prediction_enhanced_both_infinite(self) -> None:
        """Test prediction when both phi and psi are infinite."""
        # With prior, should use prior to decide
        pred, _conf = flast_enhanced._compute_prediction_enhanced(
            phi=float("inf"),
            psi=float("inf"),
            sigma=0.5,
            class_prior=0.3,
            use_prior=True,
        )
        assert pred == 0  # Prior 0.3 < sigma 0.5

        pred, _conf = flast_enhanced._compute_prediction_enhanced(
            phi=float("inf"),
            psi=float("inf"),
            sigma=0.5,
            class_prior=0.7,
            use_prior=True,
        )
        assert pred == 1  # Prior 0.7 >= sigma 0.5

    def test_compute_prediction_enhanced_phi_infinite(self) -> None:
        """Test prediction when only phi is infinite."""
        pred, conf = flast_enhanced._compute_prediction_enhanced(
            phi=float("inf"),
            psi=1.0,
            sigma=0.5,
            class_prior=0.5,
            use_prior=True,
        )
        assert pred == 1
        assert conf == 1.0

    def test_compute_prediction_enhanced_psi_infinite(self) -> None:
        """Test prediction when only psi is infinite."""
        pred, conf = flast_enhanced._compute_prediction_enhanced(
            phi=1.0,
            psi=float("inf"),
            sigma=0.5,
            class_prior=0.5,
            use_prior=True,
        )
        assert pred == 0
        assert conf == 1.0

    def test_compute_prediction_enhanced_both_zero(self) -> None:
        """Test prediction when both phi and psi are zero."""
        # With prior enabled
        pred, _conf = flast_enhanced._compute_prediction_enhanced(
            phi=0.0, psi=0.0, sigma=0.5, class_prior=0.8, use_prior=True
        )
        assert pred == 1  # High prior

        # With prior disabled
        pred, _conf = flast_enhanced._compute_prediction_enhanced(
            phi=0.0, psi=0.0, sigma=0.5, class_prior=0.8, use_prior=False
        )
        assert pred == 0  # Default to non-flaky

    def test_compute_prediction_enhanced_normal_case(self) -> None:
        """Test prediction in normal case."""
        pred, _conf = flast_enhanced._compute_prediction_enhanced(
            phi=0.8, psi=0.2, sigma=0.5, class_prior=0.5, use_prior=True
        )
        assert pred == 1  # 0.8 / 1.0 = 0.8 >= 0.5

        pred, _conf = flast_enhanced._compute_prediction_enhanced(
            phi=0.2, psi=0.8, sigma=0.5, class_prior=0.5, use_prior=True
        )
        assert pred == 0  # 0.2 / 1.0 = 0.2 < 0.5


class TestHybridFeatures:
    """Tests for hybrid features support."""

    def test_hybrid_features_dataclass(self) -> None:
        """Test HybridFeatures dataclass initialization."""
        hybrid = HybridFeatures(
            historical_flakiness_rate=[0.1, 0.2, 0.3],
            code_change_frequency=[0.5, 0.6, 0.7],
        )

        assert len(hybrid.historical_flakiness_rate) == 3
        assert len(hybrid.code_change_frequency) == 3

    def test_extract_hybrid_features_valid(self) -> None:
        """Test hybrid feature extraction with valid data."""
        hybrid = HybridFeatures(
            historical_flakiness_rate=[0.1, 0.2, 0.3],
            code_change_frequency=[0.5, 0.6, 0.7],
        )

        features = flast_enhanced.extract_hybrid_features(hybrid, n_samples=3)

        assert features is not None
        assert features.shape[0] == 3
        assert features.shape[1] == 2  # Two valid feature lists

    def test_extract_hybrid_features_empty(self) -> None:
        """Test hybrid feature extraction with empty data."""
        hybrid = HybridFeatures()
        features = flast_enhanced.extract_hybrid_features(hybrid, n_samples=3)

        assert features is None

    def test_extract_hybrid_features_none(self) -> None:
        """Test hybrid feature extraction with None."""
        features = flast_enhanced.extract_hybrid_features(None, n_samples=3)
        assert features is None

    def test_extract_hybrid_features_mismatched_length(self) -> None:
        """Test hybrid feature extraction with mismatched lengths."""
        hybrid = HybridFeatures(
            historical_flakiness_rate=[0.1, 0.2],  # Wrong length
            code_change_frequency=[0.5, 0.6, 0.7],  # Correct length
        )

        features = flast_enhanced.extract_hybrid_features(hybrid, n_samples=3)

        # Should only include the correctly-sized feature
        assert features is not None
        assert features.shape[1] == 1


class TestEnhancedFLASTClass:
    """Tests for the EnhancedFLAST class."""

    @pytest.fixture
    def sample_text_data(self) -> tuple[list[str], list[int], list[str], list[int]]:
        """Sample text data for classification."""
        train_texts = [
            "Thread sleep random timeout flaky",
            "synchronized wait notify concurrent",
            "Random nextInt shuffle UUID",
            "File Socket URL connection network",
            "assertEquals assertTrue assertNull test",
            "verify method call simple test",
            "assert equals expected actual result",
            "test data validation check simple",
        ]
        train_labels = [1, 1, 1, 1, 0, 0, 0, 0]

        test_texts = [
            "Thread.sleep timeout async wait",
            "simple test assertEquals verify",
        ]
        test_labels = [1, 0]

        return train_texts, train_labels, test_texts, test_labels

    def test_enhanced_flast_fit_predict(
        self, sample_text_data: tuple[list[str], list[int], list[str], list[int]]
    ) -> None:
        """Test fitting and predicting with EnhancedFLAST."""
        train_texts, train_labels, test_texts, _test_labels = sample_text_data

        config = EnhancedConfig(eps=0)  # No projection for small data
        clf = EnhancedFLAST(config)

        clf.fit(train_texts, train_labels)
        predictions = clf.predict(test_texts)

        assert len(predictions) == len(test_texts)
        assert all(p in [0, 1] for p in predictions)

    def test_enhanced_flast_predict_proba(
        self, sample_text_data: tuple[list[str], list[int], list[str], list[int]]
    ) -> None:
        """Test confidence scores from EnhancedFLAST."""
        train_texts, train_labels, test_texts, _ = sample_text_data

        clf = EnhancedFLAST(EnhancedConfig(eps=0))
        clf.fit(train_texts, train_labels)
        confidence = clf.predict_proba(test_texts)

        assert len(confidence) == len(test_texts)
        assert all(0 <= c <= 1 for c in confidence)

    def test_enhanced_flast_score(
        self, sample_text_data: tuple[list[str], list[int], list[str], list[int]]
    ) -> None:
        """Test scoring with EnhancedFLAST."""
        train_texts, train_labels, test_texts, test_labels = sample_text_data

        clf = EnhancedFLAST(EnhancedConfig(eps=0))
        clf.fit(train_texts, train_labels)
        precision, recall = clf.score(test_texts, test_labels)

        # Should return numeric values or "-"
        assert isinstance(precision, (float, str))
        assert isinstance(recall, (float, str))

    def test_enhanced_flast_score_extended(
        self, sample_text_data: tuple[list[str], list[int], list[str], list[int]]
    ) -> None:
        """Test extended scoring with EnhancedFLAST."""
        train_texts, train_labels, test_texts, test_labels = sample_text_data

        clf = EnhancedFLAST(EnhancedConfig(eps=0))
        clf.fit(train_texts, train_labels)
        results = clf.score_extended(test_texts, test_labels)

        assert "precision" in results
        assert "recall" in results
        assert "f1" in results
        assert "n_flaky" in results
        assert "n_non_flaky" in results

    def test_enhanced_flast_not_fitted_error(self) -> None:
        """Test error when predicting before fitting."""
        clf = EnhancedFLAST()

        with pytest.raises(RuntimeError, match="must be fitted"):
            clf.predict(["test"])

    def test_enhanced_flast_method_chaining(
        self, sample_text_data: tuple[list[str], list[int], list[str], list[int]]
    ) -> None:
        """Test method chaining with fit()."""
        train_texts, train_labels, _test_texts, _ = sample_text_data

        clf = EnhancedFLAST(EnhancedConfig(eps=0))
        result = clf.fit(train_texts, train_labels)

        assert result is clf  # fit() returns self

    def test_enhanced_flast_with_hybrid_features(
        self, sample_text_data: tuple[list[str], list[int], list[str], list[int]]
    ) -> None:
        """Test EnhancedFLAST with hybrid features."""
        train_texts, train_labels, test_texts, _test_labels = sample_text_data

        train_hybrid = HybridFeatures(
            historical_flakiness_rate=[0.8, 0.9, 0.7, 0.6, 0.1, 0.05, 0.1, 0.15],
        )
        test_hybrid = HybridFeatures(
            historical_flakiness_rate=[0.85, 0.1],
        )

        clf = EnhancedFLAST(EnhancedConfig(eps=0))
        clf.fit(train_texts, train_labels, hybrid_features=train_hybrid)
        predictions = clf.predict(test_texts, hybrid_features=test_hybrid)

        assert len(predictions) == len(test_texts)


class TestMetricsComputation:
    """Tests for extended metrics computation."""

    def test_compute_results_extended_perfect(self) -> None:
        """Test extended results with perfect predictions."""
        test_labels = [1, 1, 0, 0]
        predict_labels = [1, 1, 0, 0]

        results = compute_results_extended(test_labels, predict_labels)

        assert results["precision"] == 1.0
        assert results["recall"] == 1.0
        assert results["f1"] == 1.0
        assert results["n_flaky"] == 2
        assert results["n_non_flaky"] == 2

    def test_compute_results_extended_partial(self) -> None:
        """Test extended results with partial predictions."""
        test_labels = [1, 1, 1, 0, 0]
        predict_labels = [1, 0, 0, 0, 1]

        results = compute_results_extended(test_labels, predict_labels)

        assert results["precision"] == 0.5  # 1 TP, 1 FP
        assert results["recall"] == pytest.approx(1 / 3)  # 1 TP, 2 FN
        assert results["n_flaky"] == 3
        assert results["n_predicted_flaky"] == 2


class TestEnhancedConfig:
    """Tests for EnhancedConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = EnhancedConfig()

        assert config.use_tfidf is True
        assert config.ngram_range == (1, 2)
        assert config.use_flakiness_features is True
        assert config.use_structural_features is True
        assert config.handle_imbalance == "weights"
        assert config.use_prior_in_edge_cases is True
        assert config.k == 7
        assert config.sigma == 0.5

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = EnhancedConfig(
            use_tfidf=False,
            ngram_range=(1, 3),
            handle_imbalance="smote",
            k=5,
            sigma=0.7,
        )

        assert config.use_tfidf is False
        assert config.ngram_range == (1, 3)
        assert config.handle_imbalance == "smote"
        assert config.k == 5
        assert config.sigma == 0.7


class TestIntegration:
    """Integration tests for the full enhanced FLAST pipeline."""

    def test_full_pipeline_with_all_features(
        self, temp_dataset_dir: Path
    ) -> None:
        """Test the full enhanced pipeline with all features enabled."""
        # Load data
        flaky, non_flaky = flast_enhanced.get_data_points_info(
            temp_dataset_dir, "test-project"
        )
        all_data = flaky + non_flaky
        labels = [1] * len(flaky) + [0] * len(non_flaky)

        # Create classifier with all features
        config = EnhancedConfig(
            use_tfidf=True,
            ngram_range=(1, 2),
            use_flakiness_features=True,
            use_structural_features=True,
            handle_imbalance="weights",
            use_prior_in_edge_cases=True,
            eps=0,  # No projection for small data
            k=2,
        )

        clf = EnhancedFLAST(config)

        # Use first 3 for training, rest for testing
        train_data = all_data[:3]
        train_labels = labels[:3]
        test_data = all_data[3:]
        test_labels = labels[3:]

        clf.fit(train_data, train_labels)
        predictions = clf.predict(test_data)
        results = clf.score_extended(test_data, test_labels)

        assert len(predictions) == len(test_labels)
        assert "precision" in results
        assert "recall" in results
        assert "f1" in results
