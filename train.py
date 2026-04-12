"""Entry point CLI para treino de baselines (RandomForest e XGBoost)."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.data.loader import load_dataset
from src.evaluation.metrics import compute_metrics, plot_confusion_matrix, print_report

FIGURES_DIR = Path("reports/figures")
ARTIFACTS_DIR = Path("artifacts")


def _train_and_evaluate(
    model: object,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    le: object,
) -> dict[str, float]:
    """Treina um modelo, avalia e loga no MLflow.

    Returns:
        Dicionario de metricas.
    """
    print(f"\n{'=' * 60}")
    print(f"Treinando {model_name}...")
    print(f"{'=' * 60}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)

    print(f"\n--- {model_name} ---")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print()
    print_report(y_test, y_pred, le)

    # Confusion matrix
    fig_path = FIGURES_DIR / f"cm_{model_name.lower().replace(' ', '_')}.png"
    plot_confusion_matrix(y_test, y_pred, le, title=f"Confusion Matrix — {model_name}", output_path=fig_path)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino de baselines CICIDS2017")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Diretorio com CSVs do CICIDS2017")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fracao de teste")
    parser.add_argument("--random-state", type=int, default=42, help="Seed de reproducibilidade")
    args = parser.parse_args()

    # --- Carga dos dados ---
    print("Carregando e preparando dados...")
    X_train, X_test, y_train, y_test, le = load_dataset(
        data_dir=args.data_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape} | Classes: {len(le.classes_)}")

    num_classes = len(le.classes_)

    # --- Modelos ---
    models: dict[str, object] = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=args.random_state,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softmax",
            num_class=num_classes,
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=args.random_state,
            verbosity=0,
        ),
    }

    # --- MLflow experiment ---
    mlflow.set_experiment("cicids2017/baselines")

    results: dict[str, dict[str, float]] = {}

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("test_size", args.test_size)
            mlflow.log_param("random_state", args.random_state)
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("n_train_samples", len(X_train))

            metrics = _train_and_evaluate(model, model_name, X_train.values, y_train, X_test.values, y_test, le)
            results[model_name] = metrics

            # Log metricas
            mlflow.log_metrics(metrics)

            # Log confusion matrix como artefato
            cm_path = FIGURES_DIR / f"cm_{model_name.lower().replace(' ', '_')}.png"
            if cm_path.exists():
                mlflow.log_artifact(str(cm_path))

            # Log modelo
            if model_name == "XGBoost":
                mlflow.xgboost.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")

    # --- Selecionar melhor modelo por Macro F1 ---
    best_name = max(results, key=lambda k: results[k]["macro_f1"])
    best_model = models[best_name]
    best_metrics = results[best_name]

    print(f"\n{'=' * 60}")
    print(f"Melhor modelo: {best_name} (Macro F1 = {best_metrics['macro_f1']:.4f})")
    print(f"{'=' * 60}")

    # Salvar melhor modelo em artifacts/
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    model_path = ARTIFACTS_DIR / f"{best_name.lower()}_{timestamp}.joblib"
    joblib.dump(best_model, model_path)
    print(f"Modelo salvo em {model_path}")

    # Salvar LabelEncoder
    le_path = ARTIFACTS_DIR / f"label_encoder_{timestamp}.joblib"
    joblib.dump(le, le_path)
    print(f"LabelEncoder salvo em {le_path}")


if __name__ == "__main__":
    main()
