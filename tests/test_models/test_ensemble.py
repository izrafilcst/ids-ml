"""Testes de fumaca para src.models.ensemble (StackingEnsemble)."""

import numpy as np
import pytest
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.models.ensemble import StackingEnsemble, build_stacking_ensemble


@pytest.fixture
def synthetic_data():
    rng = np.random.RandomState(42)
    n_samples, n_features, n_classes = 300, 15, 4
    X = rng.randn(n_samples, n_features)
    y = rng.randint(0, n_classes, n_samples)
    le = LabelEncoder()
    le.fit([f"class_{i}" for i in range(n_classes)])
    return X, y, le


@pytest.fixture
def base_models(synthetic_data):
    _, y, le = synthetic_data
    n_classes = len(le.classes_)
    return {
        "RandomForest": RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=1),
        "XGBoost": XGBClassifier(
            n_estimators=10, verbosity=0, random_state=42,
            objective="multi:softmax", num_class=n_classes, tree_method="hist",
        ),
        "LightGBM": LGBMClassifier(n_estimators=10, verbose=-1, random_state=42),
    }


class TestStackingEnsemble:
    def test_fit_and_predict(self, synthetic_data, base_models):
        X, y, le = synthetic_data
        ensemble = StackingEnsemble(base_models, n_folds=2, random_state=42)
        ensemble.fit(X, y, le)
        y_pred = ensemble.predict(X)
        assert len(y_pred) == len(y)
        assert set(np.unique(y_pred)).issubset(set(np.unique(y)))

    def test_predict_proba_shape(self, synthetic_data, base_models):
        X, y, le = synthetic_data
        ensemble = StackingEnsemble(base_models, n_folds=2, random_state=42)
        ensemble.fit(X, y, le)
        proba = ensemble.predict_proba(X)
        assert proba.shape == (len(X), len(le.classes_))

    def test_proba_sums_to_one(self, synthetic_data, base_models):
        X, y, le = synthetic_data
        ensemble = StackingEnsemble(base_models, n_folds=2, random_state=42)
        ensemble.fit(X, y, le)
        proba = ensemble.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_before_fit_raises(self, base_models):
        ensemble = StackingEnsemble(base_models)
        with pytest.raises(RuntimeError, match="fit"):
            ensemble.predict(np.zeros((5, 10)))

    def test_build_stacking_ensemble_helper(self, synthetic_data, base_models):
        X, y, le = synthetic_data
        ensemble = build_stacking_ensemble(base_models, X, y, le, n_folds=2, random_state=42)
        assert isinstance(ensemble, StackingEnsemble)
        y_pred = ensemble.predict(X)
        assert len(y_pred) == len(y)
