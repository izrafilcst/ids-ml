"""Testes de fumaca para src.data.loader."""

import numpy as np
import pandas as pd
import pytest

from src.data.loader import (
    LABEL_MAP,
    clean,
    encode_labels,
    load_dataset,
    normalize_labels,
)


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """DataFrame sintetico imitando a estrutura do CICIDS2017."""
    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame(
        {
            "Flow ID": [f"id_{i}" for i in range(n)],
            "Source IP": ["192.168.1.1"] * n,
            "Destination IP": ["10.0.0.1"] * n,
            "Source Port": rng.randint(1024, 65535, n),
            "Destination Port": [80] * n,
            "Timestamp": pd.date_range("2017-07-03", periods=n, freq="s"),
            "Flow Duration": rng.randint(0, 100_000, n).astype(float),
            "Total Fwd Packets": rng.randint(0, 50, n),
            "Total Bwd Packets": rng.randint(0, 50, n),
            "Constant Col": [1] * n,
            "Label": (
                ["BENIGN"] * 140
                + ["DDoS"] * 20
                + ["PortScan"] * 15
                + ["Bot"] * 10
                + ["DoS slowloris"] * 5
                + ["Web Attack \u2013 Brute Force"] * 5
                + ["Heartbleed"] * 5
            ),
        }
    )
    # Simular valores infinitos
    df.loc[0, "Flow Duration"] = np.inf
    df.loc[1, "Flow Duration"] = -np.inf
    # Simular NaN
    df.loc[2, "Total Fwd Packets"] = np.nan
    return df


class TestClean:
    def test_removes_id_columns(self, raw_df: pd.DataFrame) -> None:
        X, _ = clean(raw_df)
        for col in ["Flow ID", "Source IP", "Destination IP", "Timestamp"]:
            assert col not in X.columns

    def test_no_infinities(self, raw_df: pd.DataFrame) -> None:
        X, _ = clean(raw_df)
        assert not np.isinf(X.values).any()

    def test_no_nans(self, raw_df: pd.DataFrame) -> None:
        X, _ = clean(raw_df)
        assert not np.isnan(X.values).any()

    def test_removes_constant_columns(self, raw_df: pd.DataFrame) -> None:
        X, _ = clean(raw_df)
        assert "Constant Col" not in X.columns

    def test_returns_labels_separately(self, raw_df: pd.DataFrame) -> None:
        _, labels = clean(raw_df)
        assert isinstance(labels, pd.Series)
        assert len(labels) > 0


class TestNormalizeLabels:
    def test_strips_whitespace(self) -> None:
        s = pd.Series(["  BENIGN ", " DDoS"])
        result = normalize_labels(s)
        assert result.iloc[0] == "Benign"
        assert result.iloc[1] == "DDoS"

    def test_maps_known_labels(self) -> None:
        for raw, expected in LABEL_MAP.items():
            result = normalize_labels(pd.Series([raw]))
            assert result.iloc[0] == expected

    def test_keeps_unknown_labels(self) -> None:
        result = normalize_labels(pd.Series(["UnknownAttack"]))
        assert result.iloc[0] == "UnknownAttack"


class TestEncodeLabels:
    def test_returns_integers(self) -> None:
        labels = pd.Series(["Benign", "DDoS", "Benign", "Bot"])
        y, le = encode_labels(labels)
        assert y.dtype in (np.int32, np.int64, int)
        assert len(le.classes_) == 3

    def test_inverse_transform(self) -> None:
        labels = pd.Series(["Benign", "DDoS", "Bot"])
        y, le = encode_labels(labels)
        recovered = le.inverse_transform(y)
        assert list(recovered) == ["Benign", "DDoS", "Bot"]


class TestLoadDataset:
    def test_with_synthetic_csvs(self, raw_df: pd.DataFrame, tmp_path) -> None:
        csv_path = tmp_path / "test.csv"
        raw_df.to_csv(csv_path, index=False)

        X_train, X_test, y_train, y_test, le = load_dataset(
            data_dir=tmp_path, test_size=0.3, random_state=42
        )

        assert len(X_train) + len(X_test) > 0
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert X_train.shape[1] == X_test.shape[1]
        assert "Label" not in X_train.columns
        assert len(le.classes_) > 1
