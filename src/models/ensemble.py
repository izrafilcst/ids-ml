"""Ensemble por stacking: RF + XGBoost + LightGBM com meta-learner.

Estrategia:
- Modelos base geram probabilidades por classe (out-of-fold no treino).
- Meta-learner (LogisticRegression) aprende a combinar essas probabilidades.
- Predição final: meta-learner sobre as probabilidades dos modelos base.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Número de folds para gerar as predições out-of-fold dos modelos base
_OOF_FOLDS = 5


class StackingEnsemble:
    """Stacking de modelos tree-based com meta-learner linear.

    Args:
        base_models: Dicionário {nome: modelo} com os modelos base já instanciados.
        meta_learner: Modelo de segundo nível. Padrão: LogisticRegression com C=1.
        n_folds: Número de folds para geração das features out-of-fold.
        random_state: Seed de reproducibilidade.
    """

    def __init__(
        self,
        base_models: dict[str, Any],
        meta_learner: Any | None = None,
        n_folds: int = _OOF_FOLDS,
        random_state: int = 42,
    ) -> None:
        self.base_models = base_models
        self.meta_learner = meta_learner or LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            n_jobs=-1,
            random_state=random_state,
        )
        self.n_folds = n_folds
        self.random_state = random_state

        self._fitted_base_models: dict[str, Any] = {}
        self._n_classes: int | None = None

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray,
        le: LabelEncoder,
    ) -> "StackingEnsemble":
        """Treina o ensemble em duas etapas.

        Etapa 1 — Out-of-Fold: cada modelo base é treinado em k-1 folds e gera
        probabilidades no fold restante. Essas probabilidades formam o conjunto
        de treino do meta-learner (sem data leakage).

        Etapa 2 — Refit completo: cada modelo base é retreinado no dataset inteiro
        para ser usado na inferência final.

        Args:
            X: Features de treino.
            y: Labels de treino (inteiros).
            le: LabelEncoder ajustado.

        Returns:
            Self, para encadeamento.
        """
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        self._n_classes = len(le.classes_)
        n_models = len(self.base_models)

        print(f"\n[Ensemble] Gerando features out-of-fold ({self.n_folds} folds, {n_models} modelos base)...")

        oof_features = np.zeros((len(X_arr), n_models * self._n_classes))
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_arr, y), 1):
            X_fold_tr, X_fold_val = X_arr[train_idx], X_arr[val_idx]
            y_fold_tr = y[train_idx]

            for model_idx, (name, model) in enumerate(self.base_models.items()):
                model.fit(X_fold_tr, y_fold_tr)
                proba = self._safe_predict_proba(model, X_fold_val)
                col_start = model_idx * self._n_classes
                col_end = col_start + self._n_classes
                oof_features[val_idx, col_start:col_end] = proba

            print(f"  Fold {fold_idx}/{self.n_folds} concluido.")

        # Etapa 2 — refit completo de cada modelo base
        print("[Ensemble] Retreinando modelos base no dataset completo...")
        for name, model in self.base_models.items():
            model.fit(X_arr, y)
            self._fitted_base_models[name] = model
            print(f"  {name} — ok.")

        # Etapa 3 — treina o meta-learner nas features OOF
        print("[Ensemble] Treinando meta-learner...")
        self.meta_learner.fit(oof_features, y)
        print("[Ensemble] Pronto.")

        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predição final via meta-learner.

        Args:
            X: Features de teste.

        Returns:
            Array de inteiros com classes preditas.
        """
        meta_features = self._build_meta_features(X)
        return self.meta_learner.predict(meta_features)

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Probabilidades finais via meta-learner.

        Args:
            X: Features de teste.

        Returns:
            Array (n_samples, n_classes) com probabilidades.
        """
        meta_features = self._build_meta_features(X)
        return self.meta_learner.predict_proba(meta_features)

    # ------------------------------------------------------------------
    # Métodos auxiliares privados
    # ------------------------------------------------------------------

    def _build_meta_features(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Constrói as features do meta-learner concatenando probabilidades dos modelos base."""
        if not self._fitted_base_models:
            raise RuntimeError("StackingEnsemble nao foi ajustado. Chame .fit() primeiro.")

        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        probas = [self._safe_predict_proba(model, X_arr) for model in self._fitted_base_models.values()]
        return np.hstack(probas)

    @staticmethod
    def _safe_predict_proba(model: Any, X: np.ndarray) -> np.ndarray:
        """Chama predict_proba; usa CalibratedClassifierCV se o modelo nao suportar."""
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        calibrated = CalibratedClassifierCV(model, cv="prefit")
        return calibrated.predict_proba(X)


# ---------------------------------------------------------------------------
# Função utilitária para construir o ensemble a partir de modelos treinados
# ---------------------------------------------------------------------------


def build_stacking_ensemble(
    base_models: dict[str, Any],
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray,
    le: LabelEncoder,
    n_folds: int = _OOF_FOLDS,
    random_state: int = 42,
) -> StackingEnsemble:
    """Instancia e treina um StackingEnsemble.

    Args:
        base_models: Dicionário {nome: modelo} já instanciados (não precisam estar treinados).
        X_train: Features de treino.
        y_train: Labels de treino (inteiros).
        le: LabelEncoder ajustado.
        n_folds: Folds para OOF.
        random_state: Seed de reproducibilidade.

    Returns:
        StackingEnsemble treinado.
    """
    ensemble = StackingEnsemble(
        base_models=base_models,
        n_folds=n_folds,
        random_state=random_state,
    )
    ensemble.fit(X_train, y_train, le)
    return ensemble


def plot_base_model_contributions(
    ensemble: StackingEnsemble,
    le: LabelEncoder,
    output_path: str | Path | None = None,
) -> None:
    """Plota a contribuicao de cada modelo base via coeficientes do meta-learner.

    Cada bloco de n_classes coeficientes corresponde a um modelo base.
    A magnitude media absoluta indica o quanto o meta-learner confia em cada modelo.

    Args:
        ensemble: StackingEnsemble ja treinado.
        le: LabelEncoder ajustado.
        output_path: Caminho para salvar o grafico.
    """
    import matplotlib.pyplot as plt

    if not hasattr(ensemble.meta_learner, "coef_"):
        print("[Ensemble] Meta-learner nao possui coef_ — plot de contribuicao indisponivel.")
        return

    coef = np.abs(ensemble.meta_learner.coef_)  # (n_classes, n_models * n_classes)
    n_classes = ensemble._n_classes or len(le.classes_)
    model_names = list(ensemble._fitted_base_models.keys())

    contributions = []
    for i, name in enumerate(model_names):
        block = coef[:, i * n_classes:(i + 1) * n_classes]
        contributions.append(block.mean())

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(model_names, contributions, color=["#4C72B0", "#DD8452", "#55A868"], edgecolor="none")

    for bar, val in zip(bars, contributions):
        ax.text(bar.get_x() + bar.get_width() / 2, val + max(contributions) * 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Coeficiente medio absoluto no meta-learner", fontsize=10)
    ax.set_title("Contribuicao de cada modelo base no Stacking", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"[Ensemble] Plot salvo em {output_path}")

    plt.close(fig)
