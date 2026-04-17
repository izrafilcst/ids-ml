"""Selecao de features baseada em SHAP values (TreeExplainer).

Uso exclusivo com modelos tree-based (RandomForest, XGBoost, LightGBM).
O seletor e serializado junto com o modelo para garantir consistencia
entre treino e inferencia.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Numero maximo de amostras para calculo de SHAP (evitar OOM em datasets grandes)
_SHAP_SAMPLE_SIZE = 10_000

# Numero padrao de features selecionadas
_DEFAULT_N_FEATURES = 40


# ---------------------------------------------------------------------------
# Classe principal
# ---------------------------------------------------------------------------


class ShapSelector:
    """Seleciona as top-k features por importancia SHAP media absoluta.

    Args:
        n_features: Numero de features a manter apos selecao.
        sample_size: Numero maximo de amostras usadas para calcular SHAP values.
            Usar um subset acelera o calculo sem perda significativa de qualidade.
        random_state: Seed para amostragem reproducivel.
    """

    def __init__(
        self,
        n_features: int = _DEFAULT_N_FEATURES,
        sample_size: int = _SHAP_SAMPLE_SIZE,
        random_state: int = 42,
    ) -> None:
        self.n_features = n_features
        self.sample_size = sample_size
        self.random_state = random_state

        self._selected_features: list[str] | None = None
        self._shap_importances: pd.Series | None = None

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------

    def fit(self, model: Any, X: pd.DataFrame | np.ndarray) -> "ShapSelector":
        """Calcula SHAP values e determina as top-k features.

        Args:
            model: Modelo tree-based ja treinado (RF, XGBoost, LightGBM).
            X: Dados de treino usados para calcular SHAP values.
                Pode ser DataFrame (preserva nomes de colunas) ou ndarray.

        Returns:
            Self, para encadeamento.
        """
        import shap

        X_df = self._to_dataframe(X)
        X_sample = self._stratified_sample(X_df)

        print(f"[ShapSelector] Calculando SHAP values em {len(X_sample):,} amostras × {X_df.shape[1]} features...")

        explainer = shap.TreeExplainer(model)

        # check_additivity=False evita falso erro de precisao em LightGBM/XGBoost
        shap_values = explainer.shap_values(X_sample, check_additivity=False)

        importances = self._aggregate_shap(shap_values, X_sample.columns.tolist())
        self._shap_importances = importances.sort_values(ascending=False)

        top_k = min(self.n_features, len(self._shap_importances))
        self._selected_features = self._shap_importances.head(top_k).index.tolist()

        print(f"[ShapSelector] {top_k} features selecionadas de {X_df.shape[1]} totais.")
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Filtra as colunas para manter apenas as features selecionadas.

        Args:
            X: Dataset a filtrar.

        Returns:
            DataFrame com apenas as features selecionadas.

        Raises:
            RuntimeError: Se ``fit`` nao foi chamado antes.
            KeyError: Se alguma feature selecionada nao existir em X.
        """
        if self._selected_features is None:
            raise RuntimeError("ShapSelector nao foi ajustado. Chame .fit() primeiro.")

        X_df = self._to_dataframe(X)
        missing = [f for f in self._selected_features if f not in X_df.columns]
        if missing:
            raise KeyError(f"Features ausentes em X: {missing}")

        return X_df[self._selected_features]

    def fit_transform(self, model: Any, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Atalho para .fit(model, X).transform(X).

        Args:
            model: Modelo tree-based ja treinado.
            X: Dados de treino.

        Returns:
            DataFrame filtrado com as top-k features.
        """
        return self.fit(model, X).transform(X)

    def plot_importance(
        self,
        output_path: str | Path | None = None,
        top_n: int | None = None,
    ) -> None:
        """Plota barchart horizontal com importancias SHAP.

        Args:
            output_path: Caminho para salvar a imagem. Se None, apenas exibe.
            top_n: Quantas features mostrar no grafico (padrao: todas as selecionadas).
        """
        if self._shap_importances is None:
            raise RuntimeError("ShapSelector nao foi ajustado. Chame .fit() primeiro.")

        import matplotlib.pyplot as plt

        n = top_n or self.n_features
        importances = self._shap_importances.head(n)

        fig, ax = plt.subplots(figsize=(10, max(6, n * 0.35)))
        bars = ax.barh(
            importances.index[::-1],
            importances.values[::-1],
            color="#4C72B0",
            edgecolor="none",
        )

        # Linha vertical no valor maximo para referencia
        ax.axvline(importances.values.max(), color="#C44E52", linewidth=0.8, linestyle="--", alpha=0.6)

        ax.set_xlabel("SHAP importance (media |valor| por feature)", fontsize=11)
        ax.set_title(f"Top {n} features por importancia SHAP", fontsize=13, fontweight="bold")
        ax.tick_params(axis="y", labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)

        # Anota valor ao lado de cada barra
        for bar, val in zip(bars, importances.values[::-1]):
            ax.text(
                val + importances.values.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                fontsize=7,
                color="#333333",
            )

        plt.tight_layout()

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150)
            print(f"[ShapSelector] Plot salvo em {output_path}")

        plt.close(fig)

    # ------------------------------------------------------------------
    # Propriedades de inspeção
    # ------------------------------------------------------------------

    @property
    def selected_features(self) -> list[str]:
        """Lista das features selecionadas apos fit."""
        if self._selected_features is None:
            raise RuntimeError("ShapSelector nao foi ajustado. Chame .fit() primeiro.")
        return list(self._selected_features)

    @property
    def importances(self) -> pd.Series:
        """Series com importancias SHAP de todas as features (ordenada decrescente)."""
        if self._shap_importances is None:
            raise RuntimeError("ShapSelector nao foi ajustado. Chame .fit() primeiro.")
        return self._shap_importances.copy()

    # ------------------------------------------------------------------
    # Metodos auxiliares privados
    # ------------------------------------------------------------------

    def _to_dataframe(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Converte ndarray para DataFrame com nomes de colunas genericos."""
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

    def _stratified_sample(self, X: pd.DataFrame) -> pd.DataFrame:
        """Amostra aleatoria simples limitada a sample_size linhas."""
        if len(X) <= self.sample_size:
            return X
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(X), size=self.sample_size, replace=False)
        return X.iloc[idx].reset_index(drop=True)

    @staticmethod
    def _aggregate_shap(
        shap_values: np.ndarray | list[np.ndarray],
        feature_names: list[str],
    ) -> pd.Series:
        """Agrega SHAP values multiclasse em importancia por feature.

        Para classificacao multiclasse, shap_values e uma lista com um array
        por classe (shape n_samples x n_features). Calcula a media das
        importancias absolutas entre todas as classes.

        Args:
            shap_values: Output do TreeExplainer.shap_values().
            feature_names: Nomes das features.

        Returns:
            Series com importancia media absoluta por feature.
        """
        sv = np.array(shap_values)
        if sv.ndim == 3:
            # Novo formato SHAP (>=0.46): (n_samples, n_features, n_classes)
            importances = np.abs(sv).mean(axis=0).mean(axis=1)
        elif isinstance(shap_values, list):
            # Formato legado: lista de (n_samples, n_features), um por classe
            stacked = np.stack([np.abs(s).mean(axis=0) for s in shap_values], axis=0)
            importances = stacked.mean(axis=0)
        else:
            # Binario: (n_samples, n_features)
            importances = np.abs(sv).mean(axis=0)

        return pd.Series(importances, index=feature_names)
