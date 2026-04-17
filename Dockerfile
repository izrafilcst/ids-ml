FROM python:3.11-slim

WORKDIR /app

# Dependências de sistema mínimas para LightGBM e compilação
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Instala dependências Python primeiro (layer cacheável)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e "." 2>/dev/null || pip install --no-cache-dir \
    "pandas>=2.2" \
    "numpy>=1.26" \
    "scikit-learn>=1.5" \
    "xgboost>=2.1" \
    "lightgbm>=4.4" \
    "imbalanced-learn>=0.12" \
    "joblib>=1.4" \
    "optuna>=3.6" \
    "shap>=0.45" \
    "matplotlib>=3.9" \
    "seaborn>=0.13" \
    "mlflow>=2.14" \
    "fastapi>=0.115" \
    "pydantic>=2.7" \
    "uvicorn>=0.30"

# Copia código fonte
COPY src/ src/
COPY predict.py .

# Diretório onde os artefatos serão montados via volume
RUN mkdir -p artifacts reports/figures

ENV ARTIFACTS_DIR=artifacts
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
