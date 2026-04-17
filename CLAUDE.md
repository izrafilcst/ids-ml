# CLAUDE.md

## Sobre o projeto
Classificador multiclasse de trafego malicioso (CICIDS2017). 15 classes: benigno + 14 ataques.
Foco: robustez, baixo falso positivo, bom recall em classes raras, pipeline reproduzivel.

## Estrutura
- `src/` — codigo de producao, dividido em data/, features/, models/, evaluation/, api/
- `notebooks/` — exploracao e prototipacao (nao sao source of truth)
- `tests/` — espelha src/
- `artifacts/` — modelos e transformers serializados (gitignored)
- `data/raw/` — CSVs originais (gitignored), `data/processed/` — dados limpos (gitignored)
- `train.py` e `predict.py` — entry points CLI
- `reports/` — plots, tabelas comparativas e relatorio final

## Convenções de codigo
- Python 3.11+
- Formatacao e lint: ruff (profile black, line-length 120)
- Type hints em todas as funcoes publicas; mypy strict nos modulos de src/
- Docstrings apenas em funcoes publicas e classes — formato Google style
- Imports agrupados: stdlib > third-party > local, gerenciado pelo ruff

## Convenções de ML
- Pipelines scikit-learn para todo pre-processamento (sem transformacao manual fora do pipeline)
- Reamostragem (SMOTE etc.) APENAS no conjunto de treino — nunca no val/teste
- Split estratificado obrigatorio (stratify=y)
- Metricas principais: Macro F1, Weighted F1, Recall por classe, Confusion Matrix
- Nao otimizar apenas accuracy — o criterio de comparacao entre modelos e Macro F1
- Modelos salvos com joblib em artifacts/ com nome: {modelo}_{timestamp}.joblib
- Hiperparametros e resultados de cada experimento logados no MLflow

### Resampling (src/features/resampling.py)
- Estrategia hibrida em 2 passos: RandomOverSampler para classes < 50 amostras, depois SMOTE para < target_minority
- target_minority padrao = 3000 amostras por classe
- k_neighbors do SMOTE ajustado automaticamente para evitar erro em classes muito raras
- Ativado via flag --resample no train.py

### Tuning (src/models/tuning.py)
- Optuna com TPESampler, 50 trials por modelo (configuravel via --n-trials)
- Usa subset estratificado de 150k amostras para velocidade
- CV de 5 folds, metrica de otimizacao: Macro F1
- Ativado via flag --tune no train.py

## Tracking de experimentos (MLflow)
- Cada run de treino deve logar em MLflow: parametros, metricas e artefato do modelo
- Tracking URI: caminho absoluto via Path(__file__).resolve().parent / "mlruns" (evita encoding de espacos no Windows)
- Experiment name segue o padrao: cicids2017/{fase}
  - cicids2017/baselines — treino sem resampling
  - cicids2017/phase2-resampling — treino com --resample
- Logar obrigatoriamente: model_type, macro_f1, weighted_f1, accuracy, confusion_matrix (como artifact)
- Modelo final registrado com mlflow.sklearn/xgboost/lightgbm.log_model conforme o tipo
- reports/ continua existindo para exports estaticos (tabelas comparativas, plots finais)

## Convenções de teste
- pytest com fixtures para datasets pequenos de teste
- Testes de fumaca para cada modelo (fit + predict em dados sinteticos)
- Testes de schema para a API (pydantic validation)

## Git
- Commits em portugues ou ingles, consistentes dentro de uma fase
- Branches: main, dev, feature/{nome}
- Nunca commitar dados (data/), artefatos (artifacts/), mlruns/ ou checkpoints

## Como rodar

```bash
# Criar venv e instalar dependencias
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"

# Treino baseline (RF + XGBoost + LightGBM)
python train.py

# Treino com resampling SMOTE nas classes minoritarias
python train.py --resample

# Treino com resampling + tuning Optuna (lento, ~1-2h)
python train.py --resample --tune --n-trials 50

# Treinar apenas modelos especificos
python train.py --models XGBoost LightGBM --resample

# Visualizar experimentos no MLflow UI
mlflow ui --backend-store-uri mlruns/
```

## Ordem de trabalho (fases)
1. [x] Ingestao + limpeza + split (src/data/loader.py)
1. [x] Baselines RF + XGBoost (train.py) — Macro F1: RF=0.86, XGB=0.89
2. [x] LightGBM + SMOTE seletivo + Optuna tuning (src/features/resampling.py, src/models/tuning.py)
2. [x] Feature selection com SHAP (src/features/selection.py)
3. [x] Ensemble final — Stacking RF + XGBoost + LightGBM (src/models/ensemble.py)
3. [x] API FastAPI + predict.py CLI (src/api/)
3. [ ] Dockerfile
3. [ ] Relatorio final
