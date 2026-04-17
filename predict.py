"""Entry point CLI para predicao em batch via linha de comando."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.api.model_loader import ModelRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="CICIDS2017 ML-IDS — predicao CLI")
    parser.add_argument("--input", type=str, required=True, help="CSV com features de entrada")
    parser.add_argument("--model", type=str, required=True, help="Caminho para o modelo .joblib")
    parser.add_argument("--encoder", type=str, required=True, help="Caminho para o LabelEncoder .joblib")
    parser.add_argument("--selector", type=str, default=None, help="Caminho para o ShapSelector .joblib (opcional)")
    parser.add_argument("--output", type=str, default=None, help="Caminho para salvar predicoes em CSV (opcional)")
    parser.add_argument("--proba", action="store_true", help="Incluir probabilidades por classe na saida")
    args = parser.parse_args()

    # Carrega artefatos
    ModelRegistry.load(args.model, args.encoder, args.selector)
    print(f"Modelo carregado: {args.model}")

    # Lê dados de entrada
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Erro: arquivo de entrada nao encontrado: {input_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(input_path, encoding="utf-8", low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"Registros carregados: {len(df):,} | Features: {df.shape[1]}")

    # Seleciona apenas as features esperadas pelo modelo (se seletor ativo)
    expected = ModelRegistry._feature_names
    if expected is not None:
        missing = [f for f in expected if f not in df.columns]
        if missing:
            print(f"Erro: features ausentes no CSV: {missing}", file=sys.stderr)
            sys.exit(1)
        X = df[expected].values.astype(np.float64)
    else:
        X = df.values.astype(np.float64)

    y_pred, proba = ModelRegistry.predict(X)
    labels = ModelRegistry.decode(y_pred)

    # Monta resultado
    result_df = df.copy()
    result_df["predicted_label"] = labels
    result_df["confidence"] = proba.max(axis=1)

    if args.proba:
        classes = ModelRegistry.classes()
        for i, cls in enumerate(classes):
            result_df[f"proba_{cls}"] = proba[:, i]

    # Resumo
    from collections import Counter
    counts = Counter(labels)
    n_attacks = sum(v for k, v in counts.items() if k != "Benign")
    print(f"\nResultados:")
    print(f"  Total de fluxos: {len(labels):,}")
    print(f"  Benigno:         {counts.get('Benign', 0):,}")
    print(f"  Ataques:         {n_attacks:,}")
    for cls, cnt in sorted(counts.items()):
        if cls != "Benign":
            print(f"    {cls:<35} {cnt:>8,}")

    # Salva ou imprime
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(out_path, index=False)
        print(f"\nPredicoes salvas em {out_path}")
    else:
        print("\nPrimeiras 10 predicoes:")
        print(result_df[["predicted_label", "confidence"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
