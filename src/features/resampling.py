"""Estrategias de reamostragem para classes minoritarias.

Resampling APENAS no conjunto de treino — nunca em val/teste.
"""

from collections import Counter

import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.preprocessing import LabelEncoder

# Classes consideradas raras no CICIDS2017 — threshold de amostras no treino
_RARE_THRESHOLD = 50
_MINORITY_THRESHOLD = 5000


def _safe_smote_k(count: int, default_k: int = 5) -> int:
    """Retorna k_neighbors seguro para SMOTE dado o numero de amostras."""
    return min(default_k, count - 1)


def build_sampling_strategy(
    y: np.ndarray,
    le: LabelEncoder,
    target_minority: int = 3000,
) -> dict[int, int]:
    """Calcula sampling_strategy para sobreamostrar apenas classes minoritarias.

    Classes com >= target_minority amostras nao sao alteradas.
    Classes muito raras (<= 50) recebem tratamento especial com RandomOverSampler.

    Args:
        y: Labels codificados (inteiros).
        le: LabelEncoder ajustado.
        target_minority: Numero alvo de amostras para classes minoritarias.

    Returns:
        Dicionario {class_idx: target_count} para classes que serao sobreamostradas.
    """
    counts = Counter(y.tolist())
    strategy: dict[int, int] = {}
    for class_idx, count in counts.items():
        if count < target_minority:
            strategy[int(class_idx)] = target_minority
    return strategy


def apply_resampling(
    X: np.ndarray,
    y: np.ndarray,
    le: LabelEncoder,
    target_minority: int = 3000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Aplica reamostragem hibrida nas classes minoritarias.

    Estrategia em dois passos:
    1. RandomOverSampler nas classes raras (< 50 amostras) para viabilizar SMOTE.
    2. SMOTE nas classes com 50..target_minority amostras.

    Args:
        X: Features do conjunto de treino.
        y: Labels do conjunto de treino (inteiros).
        le: LabelEncoder ajustado.
        target_minority: Numero alvo de amostras apos reamostragem.
        random_state: Seed de reproducibilidade.

    Returns:
        Tupla (X_resampled, y_resampled).
    """
    counts = Counter(y.tolist())

    # Passo 1 — classes raras: elevar para _RARE_THRESHOLD via RandomOverSampler
    rare_strategy: dict[int, int] = {int(idx): _RARE_THRESHOLD for idx, cnt in counts.items() if cnt < _RARE_THRESHOLD}
    if rare_strategy:
        ros = RandomOverSampler(sampling_strategy=rare_strategy, random_state=random_state)
        X, y = ros.fit_resample(X, y)
        counts = Counter(y.tolist())

    # Passo 2 — classes minoritarias: SMOTE ate target_minority
    smote_strategy: dict[int, int] = {int(idx): target_minority for idx, cnt in counts.items() if cnt < target_minority}
    if not smote_strategy:
        return X, y

    # k_neighbors seguro: limitado pelo menor numero de amostras nas classes alvo
    min_count = min(counts[idx] for idx in smote_strategy)
    k = _safe_smote_k(min_count)

    smote = SMOTE(
        sampling_strategy=smote_strategy,
        k_neighbors=k,
        random_state=random_state,
        n_jobs=-1,
    )
    X_res, y_res = smote.fit_resample(X, y)

    _print_resampling_summary(counts, Counter(y_res.tolist()), le)
    return X_res, y_res


def _print_resampling_summary(
    before: Counter,
    after: Counter,
    le: LabelEncoder,
) -> None:
    """Imprime resumo antes/depois da reamostragem."""
    print("\n--- Resampling Summary ---")
    print(f"{'Classe':<35} {'Antes':>8} {'Depois':>8} {'Delta':>8}")
    print("-" * 62)
    for idx in sorted(after.keys()):
        name = le.inverse_transform([idx])[0]
        b = before.get(idx, 0)
        a = after[idx]
        delta = f"+{a - b}" if a > b else str(a - b)
        print(f"{name:<35} {b:>8} {a:>8} {delta:>8}")
    print(f"{'TOTAL':<35} {sum(before.values()):>8} {sum(after.values()):>8}")
    print()
