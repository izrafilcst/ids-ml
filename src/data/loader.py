"""Ingestao, limpeza e split do dataset CICIDS2017."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Mapeamento de labels brutas -> labels normalizadas
# ---------------------------------------------------------------------------
LABEL_MAP: dict[str, str] = {
    "BENIGN": "Benign",
    "Bot": "Bot",
    "DDoS": "DDoS",
    "DoS GoldenEye": "DoS GoldenEye",
    "DoS Hulk": "DoS Hulk",
    "DoS Slowhttptest": "DoS Slowhttptest",
    "DoS slowloris": "DoS Slowloris",
    "FTP-Patator": "FTP-Patator",
    "Heartbleed": "Heartbleed",
    "Infiltration": "Infiltration",
    "PortScan": "PortScan",
    "SSH-Patator": "SSH-Patator",
    "Web Attack \u2013 Brute Force": "Web Attack Brute Force",
    "Web Attack \u2013 XSS": "Web Attack XSS",
    "Web Attack \u2013 Sql Injection": "Web Attack Sql Injection",
}

# Colunas que vazam informacao ou sao identificadores inuteis
COLUMNS_TO_DROP: list[str] = [
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "Timestamp",
]


def load_csvs(data_dir: str | Path = "data/raw") -> pd.DataFrame:
    """Carrega e concatena todos os CSVs do CICIDS2017.

    Args:
        data_dir: Diretorio contendo os arquivos CSV.

    Returns:
        DataFrame unico com todos os registros.

    Raises:
        FileNotFoundError: Se nenhum CSV for encontrado em ``data_dir``.
    """
    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {data_dir.resolve()}")

    frames: list[pd.DataFrame] = []
    for f in csv_files:
        df = pd.read_csv(f, encoding="utf-8", low_memory=False)
        df.columns = df.columns.str.strip()
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Limpeza basica: remove ids, trata inf/NaN e colunas constantes.

    Args:
        df: DataFrame bruto carregado por :func:`load_csvs`.

    Returns:
        DataFrame limpo, pronto para modelagem.
    """
    # Remover colunas de identificacao (ignora se nao existirem)
    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Separar label antes de mexer nas features
    label_col = "Label"
    labels = df[label_col].copy()
    df = df.drop(columns=[label_col])

    # Converter tudo para numerico (colunas que falharem viram NaN)
    df = df.apply(pd.to_numeric, errors="coerce")

    # Substituir infinitos por NaN e depois preencher com a mediana da coluna
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)

    # Remover colunas constantes (variancia zero)
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    df = df.drop(columns=constant_cols)

    # Remover duplicatas
    df[label_col] = labels
    df = df.drop_duplicates()
    labels = df[label_col]
    df = df.drop(columns=[label_col])

    return df, labels


def normalize_labels(labels: pd.Series) -> pd.Series:
    """Normaliza as labels brutas usando LABEL_MAP.

    Labels que nao estiverem no mapa sao mantidas com strip().

    Args:
        labels: Series com labels brutas do dataset.

    Returns:
        Series com labels normalizadas.
    """
    labels = labels.str.strip()
    return labels.map(lambda x: LABEL_MAP.get(x, x))


def encode_labels(labels: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    """Codifica labels categoricas em inteiros.

    Args:
        labels: Series com labels normalizadas.

    Returns:
        Tupla (array de inteiros, LabelEncoder ajustado).
    """
    le = LabelEncoder()
    y = le.fit_transform(labels)
    return y, le


def load_dataset(
    data_dir: str | Path = "data/raw",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, LabelEncoder]:
    """Pipeline completo: carga, limpeza, encoding e split estratificado.

    Args:
        data_dir: Diretorio com os CSVs do CICIDS2017.
        test_size: Fracao do dataset reservada para teste.
        random_state: Seed para reprodutibilidade.

    Returns:
        Tupla (X_train, X_test, y_train, y_test, label_encoder).
    """
    df = load_csvs(data_dir)
    X, labels = clean(df)
    labels = normalize_labels(labels)
    y, le = encode_labels(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test, le
