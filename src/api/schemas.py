"""Schemas Pydantic para request e response da API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class FlowFeatures(BaseModel):
    """Features de um fluxo de rede para classificacao.

    Todos os campos correspondem às features do CICFlowMeter apos limpeza.
    Valores infinitos ou NaN nao sao aceitos — pre-processe antes de enviar.
    """

    model_config = {"json_schema_extra": {
        "example": {
            "features": {
                "Flow Duration": 1234567,
                "Total Fwd Packets": 10,
                "Total Backward Packets": 8,
                "Total Length of Fwd Packets": 1500,
                "Total Length of Bwd Packets": 900,
                "Fwd Packet Length Max": 200,
                "Fwd Packet Length Min": 40,
                "Fwd Packet Length Mean": 150.0,
                "Fwd Packet Length Std": 30.5,
                "Bwd Packet Length Max": 180,
                "Bwd Packet Length Min": 20,
                "Bwd Packet Length Mean": 112.5,
                "Bwd Packet Length Std": 25.0,
                "Flow Bytes/s": 12345.6,
                "Flow Packets/s": 15.2,
            }
        }
    }}

    features: dict[str, float] = Field(
        ...,
        description="Dicionario com nome_da_feature → valor numerico.",
    )

    @field_validator("features")
    @classmethod
    def no_nan_or_inf(cls, v: dict[str, float]) -> dict[str, float]:
        import math
        bad = [k for k, val in v.items() if not math.isfinite(val)]
        if bad:
            raise ValueError(f"Valores nao finitos detectados nas features: {bad}")
        return v


class BatchFlowFeatures(BaseModel):
    """Batch de fluxos para classificacao em lote."""

    flows: list[FlowFeatures] = Field(..., min_length=1, max_length=10_000)


class PredictionResult(BaseModel):
    """Resultado de classificacao para um unico fluxo."""

    label: str = Field(..., description="Classe predita (ex: 'Benign', 'DDoS').")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Probabilidade maxima entre as classes.")
    probabilities: dict[str, float] = Field(..., description="Probabilidade de cada classe.")


class BatchPredictionResult(BaseModel):
    """Resultado de classificacao em lote."""

    predictions: list[PredictionResult]
    n_flows: int
    n_attacks: int
    attack_breakdown: dict[str, int] = Field(
        ..., description="Contagem por tipo de ataque detectado (exclui Benign)."
    )


class HealthResponse(BaseModel):
    """Resposta do endpoint de health check."""

    status: str
    model_loaded: bool
    model_path: str | None
    n_features: int | None
    classes: list[str] | None
