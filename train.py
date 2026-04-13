"""Entry point CLI para treino de baselines e modelos otimizados (Phase 1 & 2)."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

import joblib
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.data.loader import load_dataset
from src.evaluation.metrics import compute_metrics, plot_confusion_matrix, print_report
from src.features.resampling import apply_resampling

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
    """Treina um modelo, avalia no holdout e retorna metricas.

    Returns:
        Dicionario com accuracy, macro_f1, weighted_f1.
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

    fig_path = FIGURES_DIR / f"cm_{model_name.lower().replace(' ', '_')}.png"
    plot_confusion_matrix(y_test, y_pred, le, title=f"Confusion Matrix — {model_name}", output_path=fig_path)

    return metrics


def _log_model_mlflow(model: object, model_name: str) -> None:
    """Loga modelo no MLflow com o flavour correto."""
    if model_name == "XGBoost":
        mlflow.xgboost.log_model(model, name="model")
    elif model_name == "LightGBM":
        mlflow.lightgbm.log_model(model, name="model")
    else:
        mlflow.sklearn.log_model(model, name="model")


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino CICIDS2017 — Phase 1 & 2")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Diretorio com CSVs do CICIDS2017")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fracao de teste")
    parser.add_argument("--random-state", type=int, default=42, help="Seed de reproducibilidade")
    parser.add_argument(
        "--resample",
        action="store_true",
        help="Aplica SMOTE seletivo nas classes minoritarias (apenas no treino)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Executa tuning de hiperparametros com Optuna antes de treinar (lento)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Numero de trials Optuna por modelo (default: 50)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["RandomForest", "XGBoost", "LightGBM"],
        choices=["RandomForest", "XGBoost", "LightGBM"],
        help="Modelos a treinar",
    )
    args = parser.parse_args()

    # --- Carga dos dados ---
    print("Carregando e preparando dados...")
    X_train, X_test, y_train, y_test, le = load_dataset(
        data_dir=args.data_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape} | Classes: {len(le.classes_)}")

    X_tr = X_train.values
    X_te = X_test.values

    # --- Resampling (somente no treino) ---
    if args.resample:
        print("\nAplicando resampling nas classes minoritarias...")
        X_tr, y_train = apply_resampling(X_tr, y_train, le, random_state=args.random_state)
        print(f"Train apos resampling: {X_tr.shape}")

    num_classes = len(le.classes_)

    # --- Hiperparametros (defaults ou tunados) ---
    xgb_params: dict = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "objective": "multi:softmax",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": args.random_state,
        "verbosity": 0,
    }
    lgbm_params: dict = {
        "n_estimators": 200,
        "num_leaves": 63,
        "max_depth": -1,
        "learning_rate": 0.1,
        "class_weight": "balanced",
        "objective": "multiclass",
        "n_jobs": -1,
        "random_state": args.random_state,
        "verbose": -1,
    }

    if args.tune:
        from src.models.tuning import tune_lightgbm, tune_xgboost

        print("\nIniciando tuning com Optuna...")
        if "XGBoost" in args.models:
            best = tune_xgboost(X_tr, y_train, le, n_trials=args.n_trials, random_state=args.random_state)
            xgb_params.update(best)
            # garantir parametros fixos obrigatorios
            xgb_params.update({"objective": "multi:softmax", "num_class": num_classes, "tree_method": "hist"})
        if "LightGBM" in args.models:
            best = tune_lightgbm(X_tr, y_train, le, n_trials=args.n_trials, random_state=args.random_state)
            lgbm_params.update(best)
            lgbm_params.update({"objective": "multiclass"})

    # --- Definicao dos modelos ---
    all_models: dict[str, object] = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=args.random_state,
        ),
        "XGBoost": XGBClassifier(**xgb_params),
        "LightGBM": LGBMClassifier(**lgbm_params),
    }
    models = {name: mdl for name, mdl in all_models.items() if name in args.models}

    # --- MLflow experiment ---
    mlflow_dir = Path(__file__).resolve().parent / "mlruns"
    mlflow.set_tracking_uri(mlflow_dir.as_uri())
    experiment_name = "cicids2017/phase2-resampling" if args.resample else "cicids2017/baselines"
    mlflow.set_experiment(experiment_name)

    results: dict[str, dict[str, float]] = {}

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("test_size", args.test_size)
            mlflow.log_param("random_state", args.random_state)
            mlflow.log_param("resampling", args.resample)
            mlflow.log_param("tuned", args.tune)
            mlflow.log_param("n_features", X_tr.shape[1])
            mlflow.log_param("n_train_samples", len(X_tr))

            metrics = _train_and_evaluate(model, model_name, X_tr, y_train, X_te, y_test, le)
            results[model_name] = metrics

            mlflow.log_metrics(metrics)

            cm_path = FIGURES_DIR / f"cm_{model_name.lower().replace(' ', '_')}.png"
            if cm_path.exists():
                mlflow.log_artifact(str(cm_path))

            _log_model_mlflow(model, model_name)

    # --- Melhor modelo por Macro F1 ---
    best_name = max(results, key=lambda k: results[k]["macro_f1"])
    best_model = models[best_name]
    best_metrics = results[best_name]

    print(f"\n{'=' * 60}")
    print(f"Melhor modelo: {best_name} (Macro F1 = {best_metrics['macro_f1']:.4f})")
    print(f"{'=' * 60}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")

    suffix = "_resampled" if args.resample else ""
    model_path = ARTIFACTS_DIR / f"{best_name.lower()}{suffix}_{timestamp}.joblib"
    joblib.dump(best_model, model_path)
    print(f"Modelo salvo em {model_path}")

    le_path = ARTIFACTS_DIR / f"label_encoder_{timestamp}.joblib"
    joblib.dump(le, le_path)
    print(f"LabelEncoder salvo em {le_path}")


if __name__ == "__main__":
    main()
