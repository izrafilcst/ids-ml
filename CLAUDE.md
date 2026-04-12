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

## Tracking de experimentos (MLflow)
- Cada run de treino deve logar em MLflow: parametros, metricas e artefato do modelo
- Tracking URI padrao: local (./mlruns) — sem servidor externo na V1
- Experiment name segue o padrao: cicids2017/{fase} (ex: cicids2017/baselines, cicids2017/boosting)
- Logar obrigatoriamente: model_type, macro_f1, weighted_f1, accuracy, confusion_matrix (como artifact)
- Modelo final registrado com mlflow.sklearn.log_model ou mlflow.xgboost.log_model
- reports/ continua existindo para exports estaticos (tabelas comparativas, plots finais)

## Convenções de teste
- pytest com fixtures para datasets pequenos de teste
- Testes de fumaca para cada modelo (fit + predict em dados sinteticos)
- Testes de schema para a API (pydantic validation)

## Git
- Commits em portugues ou ingles, consistentes dentro de uma fase
- Branches: main, dev, feature/{nome}
- Nunca commitar dados (data/), artefatos (artifacts/), mlruns/ ou checkpoints

## Ordem de trabalho (fases)
1. Ingestao + limpeza + split -> baselines (LR, DT, RF)
2. XGBoost + LightGBM + feature selection + resampling + tuning
3. Ensemble final + API FastAPI + Dockerfile + relatorio
