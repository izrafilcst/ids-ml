"""Tuning de hiperparametros com Optuna para XGBoost e LightGBM."""

from __future__ import annotations

import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Numero de amostras para tuning — usa subset estratificado para velocidade
_TUNE_SAMPLE_SIZE = 150_000
_CV_FOLDS = 5


def _stratified_sample(
    X: np.ndarray,
    y: np.ndarray,
    n: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Amostra estratificada de n linhas preservando distribuicao de classes."""
    from sklearn.model_selection import train_test_split

    if len(X) <= n:
        return X, y
    _, X_s, _, y_s = train_test_split(X, y, test_size=n / len(X), stratify=y, random_state=random_state)
    return X_s, y_s


def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    le: LabelEncoder,
    n_trials: int = 50,
    random_state: int = 42,
) -> dict:
    """Busca hiperparametros otimos para XGBoost via Optuna.

    Usa subset estratificado de _TUNE_SAMPLE_SIZE linhas e CV de _CV_FOLDS folds.
    Metrica de otimizacao: Macro F1.

    Args:
        X_train: Features de treino.
        y_train: Labels de treino (inteiros).
        le: LabelEncoder ajustado.
        n_trials: Numero de trials Optuna.
        random_state: Seed de reproducibilidade.

    Returns:
        Dicionario com os melhores hiperparametros encontrados.
    """
    X_s, y_s = _stratified_sample(X_train, y_train, _TUNE_SAMPLE_SIZE, random_state)
    num_classes = len(le.classes_)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "objective": "multi:softmax",
            "num_class": num_classes,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": random_state,
            "verbosity": 0,
        }
        model = XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=_CV_FOLDS, shuffle=True, random_state=random_state)
        scores = cross_val_score(model, X_s, y_s, cv=cv, scoring="f1_macro", n_jobs=1)
        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n[XGBoost] Melhor Macro F1 (CV): {study.best_value:.4f}")
    print(f"[XGBoost] Melhores params: {study.best_params}")
    return study.best_params


def tune_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    le: LabelEncoder,
    n_trials: int = 50,
    random_state: int = 42,
) -> dict:
    """Busca hiperparametros otimos para LightGBM via Optuna.

    Args:
        X_train: Features de treino.
        y_train: Labels de treino (inteiros).
        le: LabelEncoder ajustado.
        n_trials: Numero de trials Optuna.
        random_state: Seed de reproducibilidade.

    Returns:
        Dicionario com os melhores hiperparametros encontrados.
    """
    X_s, y_s = _stratified_sample(X_train, y_train, _TUNE_SAMPLE_SIZE, random_state)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "class_weight": "balanced",
            "objective": "multiclass",
            "n_jobs": -1,
            "random_state": random_state,
            "verbose": -1,
        }
        model = LGBMClassifier(**params)
        cv = StratifiedKFold(n_splits=_CV_FOLDS, shuffle=True, random_state=random_state)
        scores = cross_val_score(model, X_s, y_s, cv=cv, scoring="f1_macro", n_jobs=1)
        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n[LightGBM] Melhor Macro F1 (CV): {study.best_value:.4f}")
    print(f"[LightGBM] Melhores params: {study.best_params}")
    return study.best_params
