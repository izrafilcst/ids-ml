"""Testes de fumaca para src.features.resampling."""

import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder

from src.features.resampling import apply_resampling, build_sampling_strategy


@pytest.fixture
def imbalanced_data():
    rng = np.random.RandomState(42)
    # 3 classes: majoritaria (1000), minoritaria (200), rara (30)
    X = rng.randn(1230, 10)
    y = np.array([0] * 1000 + [1] * 200 + [2] * 30)
    le = LabelEncoder()
    le.fit(["Benign", "DDoS", "Heartbleed"])
    return X, y, le


class TestBuildSamplingStrategy:
    def test_skips_majority_class(self, imbalanced_data):
        X, y, le = imbalanced_data
        strategy = build_sampling_strategy(y, le, target_minority=500)
        assert 0 not in strategy  # classe majoritaria nao e sobreamostrada

    def test_includes_minority_classes(self, imbalanced_data):
        X, y, le = imbalanced_data
        strategy = build_sampling_strategy(y, le, target_minority=500)
        assert 1 in strategy
        assert 2 in strategy

    def test_target_respected(self, imbalanced_data):
        X, y, le = imbalanced_data
        strategy = build_sampling_strategy(y, le, target_minority=500)
        assert strategy[1] == 500
        assert strategy[2] == 500


class TestApplyResampling:
    def test_output_shape_increases(self, imbalanced_data):
        X, y, le = imbalanced_data
        X_res, y_res = apply_resampling(X, y, le, target_minority=500)
        assert len(X_res) > len(X)
        assert len(X_res) == len(y_res)

    def test_n_features_unchanged(self, imbalanced_data):
        X, y, le = imbalanced_data
        X_res, y_res = apply_resampling(X, y, le, target_minority=500)
        assert X_res.shape[1] == X.shape[1]

    def test_all_classes_present(self, imbalanced_data):
        X, y, le = imbalanced_data
        X_res, y_res = apply_resampling(X, y, le, target_minority=500)
        assert set(np.unique(y_res)) == {0, 1, 2}

    def test_majority_class_untouched(self, imbalanced_data):
        X, y, le = imbalanced_data
        X_res, y_res = apply_resampling(X, y, le, target_minority=500)
        # Classe majoritaria (1000) nao deve ser reduzida
        assert np.sum(y_res == 0) >= 1000
