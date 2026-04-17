"""API FastAPI para classificacao de trafego de rede (CICIDS2017)."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from collections import Counter

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from src.api.model_loader import ModelRegistry, autoload
from src.api.schemas import (
    BatchFlowFeatures,
    BatchPredictionResult,
    FlowFeatures,
    HealthResponse,
    PredictionResult,
)

# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifacts_dir = os.environ.get("ARTIFACTS_DIR", "artifacts")
    loaded = autoload(artifacts_dir)
    if not loaded:
        print(
            f"[API] Aviso: nenhum modelo encontrado em '{artifacts_dir}'. "
            "Use os endpoints /predict somente apos carregar um modelo."
        )
    else:
        print(f"[API] Modelo carregado: {ModelRegistry._model_path}")
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ML-IDS — CICIDS2017 Classifier",
    description=(
        "API de classificacao multiclasse de trafego de rede. "
        "Detecta 14 tipos de ataque + trafego benigno com base no dataset CIC-IDS-2017."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["Infra"])
def health() -> HealthResponse:
    """Verifica se a API está operacional e se o modelo está carregado."""
    return HealthResponse(
        status="ok" if ModelRegistry.is_loaded() else "degraded",
        model_loaded=ModelRegistry.is_loaded(),
        model_path=ModelRegistry._model_path,
        n_features=ModelRegistry.n_features(),
        classes=ModelRegistry.classes() or None,
    )


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------


def _features_to_array(flow: FlowFeatures) -> np.ndarray:
    """Converte FlowFeatures em array numpy na ordem correta."""
    expected = ModelRegistry._feature_names
    if expected is not None:
        missing = [f for f in expected if f not in flow.features]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Features obrigatórias ausentes: {missing}",
            )
        return np.array([[flow.features[f] for f in expected]], dtype=np.float64)

    # Sem seletor: usa a ordem recebida (todos os campos)
    return np.array([list(flow.features.values())], dtype=np.float64)


def _build_prediction(y_pred: np.ndarray, proba: np.ndarray) -> PredictionResult:
    labels = ModelRegistry.decode(y_pred)
    classes = ModelRegistry.classes()
    prob_dict = {cls: float(p) for cls, p in zip(classes, proba[0])}
    return PredictionResult(
        label=labels[0],
        confidence=float(proba[0].max()),
        probabilities=prob_dict,
    )


# ---------------------------------------------------------------------------
# Endpoints de predição
# ---------------------------------------------------------------------------


@app.post("/predict", response_model=PredictionResult, tags=["Classificação"])
def predict(flow: FlowFeatures) -> PredictionResult:
    """Classifica um único fluxo de rede.

    Retorna a classe predita, confiança e probabilidades por classe.
    """
    if not ModelRegistry.is_loaded():
        raise HTTPException(status_code=503, detail="Modelo não carregado.")

    X = _features_to_array(flow)
    y_pred, proba = ModelRegistry.predict(X)
    return _build_prediction(y_pred, proba)


@app.post("/predict/batch", response_model=BatchPredictionResult, tags=["Classificação"])
def predict_batch(batch: BatchFlowFeatures) -> BatchPredictionResult:
    """Classifica um lote de fluxos de rede (máx. 10.000).

    Retorna predições individuais + resumo com contagem de ataques detectados.
    """
    if not ModelRegistry.is_loaded():
        raise HTTPException(status_code=503, detail="Modelo não carregado.")

    rows = []
    for flow in batch.flows:
        X = _features_to_array(flow)
        rows.append(X[0])

    X_batch = np.array(rows, dtype=np.float64)
    y_pred, proba = ModelRegistry.predict(X_batch)
    labels = ModelRegistry.decode(y_pred)
    classes = ModelRegistry.classes()

    predictions = [
        PredictionResult(
            label=label,
            confidence=float(p.max()),
            probabilities={cls: float(v) for cls, v in zip(classes, p)},
        )
        for label, p in zip(labels, proba)
    ]

    label_counts = Counter(labels)
    attack_breakdown = {k: v for k, v in label_counts.items() if k != "Benign"}

    return BatchPredictionResult(
        predictions=predictions,
        n_flows=len(predictions),
        n_attacks=sum(attack_breakdown.values()),
        attack_breakdown=attack_breakdown,
    )


@app.get("/classes", tags=["Infra"])
def list_classes() -> JSONResponse:
    """Lista todas as classes que o modelo consegue predizer."""
    if not ModelRegistry.is_loaded():
        raise HTTPException(status_code=503, detail="Modelo não carregado.")
    return JSONResponse({"classes": ModelRegistry.classes(), "n_classes": len(ModelRegistry.classes())})
