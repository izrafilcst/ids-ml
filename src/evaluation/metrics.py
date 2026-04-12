"""Funcoes de avaliacao e visualizacao de resultados."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula metricas de classificacao.

    Args:
        y_true: Labels verdadeiras (inteiros).
        y_pred: Labels preditas (inteiros).

    Returns:
        Dicionario com accuracy, macro_f1 e weighted_f1.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def print_report(y_true: np.ndarray, y_pred: np.ndarray, le: LabelEncoder) -> str:
    """Gera e imprime o classification_report completo.

    Args:
        y_true: Labels verdadeiras (inteiros).
        y_pred: Labels preditas (inteiros).
        le: LabelEncoder ajustado para converter indices em nomes.

    Returns:
        Texto do classification_report.
    """
    target_names = list(le.classes_)
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    print(report)
    return report


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    le: LabelEncoder,
    title: str = "Confusion Matrix",
    output_path: str | Path | None = None,
) -> None:
    """Plota e opcionalmente salva a confusion matrix.

    Args:
        y_true: Labels verdadeiras (inteiros).
        y_pred: Labels preditas (inteiros).
        le: LabelEncoder ajustado.
        title: Titulo do grafico.
        output_path: Caminho para salvar a imagem. Se None, apenas exibe.
    """
    labels = list(le.classes_)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(max(10, len(labels)), max(8, len(labels) * 0.8)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"Confusion matrix salva em {output_path}")

    plt.close(fig)
