"""Carregamento e cache do modelo, LabelEncoder e ShapSelector."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np


class ModelRegistry:
    """Singleton que mantém modelo, encoder e seletor em memória.

    Carregado uma vez no startup da API e reutilizado em todas as requests.
    """

    _model: object | None = None
    _le: object | None = None
    _selector: object | None = None
    _model_path: str | None = None
    _feature_names: list[str] | None = None

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        le_path: str | Path,
        selector_path: str | Path | None = None,
    ) -> None:
        """Carrega artefatos do disco para memória.

        Args:
            model_path: Caminho para o modelo serializado (.joblib).
            le_path: Caminho para o LabelEncoder serializado (.joblib).
            selector_path: Caminho para o ShapSelector serializado (.joblib).
                           Opcional — só necessário se o modelo foi treinado com --select.
        """
        cls._model = joblib.load(model_path)
        cls._le = joblib.load(le_path)
        cls._model_path = str(model_path)

        if selector_path is not None and Path(selector_path).exists():
            cls._selector = joblib.load(selector_path)
            cls._feature_names = cls._selector.selected_features
        else:
            cls._selector = None
            cls._feature_names = None

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._model is not None and cls._le is not None

    @classmethod
    def predict(cls, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Executa predição e retorna (labels preditas, matriz de probabilidades).

        Args:
            X: Array (n_samples, n_features) com os dados já ordenados.

        Returns:
            Tupla (y_pred inteiros, proba array n_samples × n_classes).
        """
        if not cls.is_loaded():
            raise RuntimeError("Modelo nao carregado. Chame ModelRegistry.load() no startup.")

        if cls._selector is not None:
            import pandas as pd
            X = cls._selector.transform(pd.DataFrame(X, columns=cls._selector.selected_features))
            X = X.values

        y_pred = cls._model.predict(X)

        if hasattr(cls._model, "predict_proba"):
            proba = cls._model.predict_proba(X)
        else:
            # Fallback: one-hot a partir da predição
            n_classes = len(cls._le.classes_)
            proba = np.zeros((len(y_pred), n_classes))
            proba[np.arange(len(y_pred)), y_pred] = 1.0

        return y_pred, proba

    @classmethod
    def decode(cls, y: np.ndarray) -> list[str]:
        """Converte inteiros para nomes de classes."""
        return list(cls._le.inverse_transform(y))

    @classmethod
    def classes(cls) -> list[str]:
        return list(cls._le.classes_) if cls._le is not None else []

    @classmethod
    def n_features(cls) -> int | None:
        return len(cls._feature_names) if cls._feature_names else None


def _latest_artifact(artifacts_dir: Path, pattern: str) -> Path | None:
    """Retorna o artefato mais recente que casa com o glob pattern."""
    matches = sorted(artifacts_dir.glob(pattern))
    return matches[-1] if matches else None


def autoload(artifacts_dir: str | Path = "artifacts") -> bool:
    """Tenta carregar automaticamente o modelo mais recente em artifacts/.

    Ordem de preferência: modelo com '_selected', depois qualquer outro.
    Retorna True se carregou com sucesso, False caso contrário.
    """
    base = Path(artifacts_dir)
    if not base.exists():
        return False

    # Modelo: preferência para versão com seleção de features
    model_path = _latest_artifact(base, "*selected*.joblib")
    if model_path is None:
        # Exclui label_encoder e shap_selector da busca
        candidates = [p for p in sorted(base.glob("*.joblib"))
                      if "label_encoder" not in p.name and "shap_selector" not in p.name]
        model_path = candidates[-1] if candidates else None

    le_path = _latest_artifact(base, "label_encoder_*.joblib")
    selector_path = _latest_artifact(base, "shap_selector_*.joblib")

    if model_path is None or le_path is None:
        return False

    ModelRegistry.load(model_path, le_path, selector_path)
    return True
