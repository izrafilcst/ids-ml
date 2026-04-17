"""Testes de fumaca para src.features.selection (ShapSelector)."""

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier

from src.features.selection import ShapSelector


@pytest.fixture
def trained_model_and_data():
    rng = np.random.RandomState(42)
    n_samples, n_features = 300, 20
    X = pd.DataFrame(rng.randn(n_samples, n_features), columns=[f"feat_{i}" for i in range(n_features)])
    y = rng.randint(0, 3, n_samples)
    model = LGBMClassifier(n_estimators=20, verbose=-1, random_state=42)
    model.fit(X, y)
    return model, X, y


class TestShapSelector:
    def test_fit_returns_self(self, trained_model_and_data):
        model, X, _ = trained_model_and_data
        selector = ShapSelector(n_features=10)
        result = selector.fit(model, X)
        assert result is selector

    def test_selected_features_count(self, trained_model_and_data):
        model, X, _ = trained_model_and_data
        selector = ShapSelector(n_features=10)
        selector.fit(model, X)
        assert len(selector.selected_features) == 10

    def test_selected_features_subset_of_columns(self, trained_model_and_data):
        model, X, _ = trained_model_and_data
        selector = ShapSelector(n_features=10)
        selector.fit(model, X)
        assert all(f in X.columns for f in selector.selected_features)

    def test_transform_reduces_columns(self, trained_model_and_data):
        model, X, _ = trained_model_and_data
        selector = ShapSelector(n_features=10)
        selector.fit(model, X)
        X_sel = selector.transform(X)
        assert X_sel.shape[1] == 10
        assert X_sel.shape[0] == len(X)

    def test_fit_transform_consistent(self, trained_model_and_data):
        model, X, _ = trained_model_and_data
        selector = ShapSelector(n_features=8)
        X1 = selector.fit_transform(model, X)
        X2 = selector.transform(X)
        pd.testing.assert_frame_equal(X1, X2)

    def test_transform_before_fit_raises(self):
        selector = ShapSelector(n_features=5)
        with pytest.raises(RuntimeError, match="fit"):
            selector.transform(pd.DataFrame([[1, 2, 3]]))

    def test_importances_sorted_descending(self, trained_model_and_data):
        model, X, _ = trained_model_and_data
        selector = ShapSelector(n_features=10)
        selector.fit(model, X)
        vals = selector.importances.values
        assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))

    def test_n_features_capped_at_total(self, trained_model_and_data):
        model, X, _ = trained_model_and_data
        selector = ShapSelector(n_features=999)
        selector.fit(model, X)
        assert len(selector.selected_features) == X.shape[1]
