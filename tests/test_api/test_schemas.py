"""Testes de schema Pydantic para a API."""

import math

import pytest
from pydantic import ValidationError

from src.api.schemas import (
    BatchFlowFeatures,
    BatchPredictionResult,
    FlowFeatures,
    HealthResponse,
    PredictionResult,
)


VALID_FEATURES = {"Flow Duration": 1234567.0, "Total Fwd Packets": 10.0, "Total Bwd Packets": 8.0}


class TestFlowFeatures:
    def test_valid_features(self):
        flow = FlowFeatures(features=VALID_FEATURES)
        assert flow.features["Flow Duration"] == 1234567.0

    def test_rejects_nan(self):
        with pytest.raises(ValidationError, match="nao finitos"):
            FlowFeatures(features={"f1": float("nan")})

    def test_rejects_inf(self):
        with pytest.raises(ValidationError, match="nao finitos"):
            FlowFeatures(features={"f1": float("inf")})

    def test_rejects_neg_inf(self):
        with pytest.raises(ValidationError, match="nao finitos"):
            FlowFeatures(features={"f1": float("-inf")})

    def test_empty_features_accepted(self):
        flow = FlowFeatures(features={})
        assert flow.features == {}


class TestBatchFlowFeatures:
    def test_valid_batch(self):
        batch = BatchFlowFeatures(flows=[FlowFeatures(features=VALID_FEATURES)] * 3)
        assert len(batch.flows) == 3

    def test_rejects_empty_batch(self):
        with pytest.raises(ValidationError):
            BatchFlowFeatures(flows=[])


class TestPredictionResult:
    def test_valid(self):
        result = PredictionResult(
            label="DDoS",
            confidence=0.95,
            probabilities={"Benign": 0.05, "DDoS": 0.95},
        )
        assert result.label == "DDoS"
        assert result.confidence == 0.95

    def test_confidence_out_of_range(self):
        with pytest.raises(ValidationError):
            PredictionResult(label="X", confidence=1.5, probabilities={})


class TestBatchPredictionResult:
    def test_valid(self):
        pred = PredictionResult(label="Benign", confidence=0.99, probabilities={"Benign": 0.99})
        result = BatchPredictionResult(
            predictions=[pred],
            n_flows=1,
            n_attacks=0,
            attack_breakdown={},
        )
        assert result.n_flows == 1
        assert result.n_attacks == 0


class TestHealthResponse:
    def test_loaded(self):
        h = HealthResponse(
            status="ok",
            model_loaded=True,
            model_path="artifacts/model.joblib",
            n_features=40,
            classes=["Benign", "DDoS"],
        )
        assert h.status == "ok"

    def test_degraded(self):
        h = HealthResponse(
            status="degraded",
            model_loaded=False,
            model_path=None,
            n_features=None,
            classes=None,
        )
        assert not h.model_loaded
