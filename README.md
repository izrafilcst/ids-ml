# CICIDS2017 — ML-IDS

> **Classificador multiclasse de tráfego de rede** baseado no dataset CIC-IDS-2017.
> Detecta 14 tipos de ataque + tráfego benigno com foco em baixo falso positivo e alto recall em classes raras.

---

## Resultados

| Modelo | Macro F1 | Weighted F1 | Accuracy |
|---|---|---|---|
| RandomForest (baseline) | 0.86 | — | — |
| XGBoost (baseline) | 0.89 | — | — |
| LightGBM + SMOTE + Optuna | *em andamento* | — | — |

> Critério de comparação entre modelos: **Macro F1** (penaliza classes minoritárias não detectadas).

---

## Classes detectadas (15)

```
Benign · Bot · DDoS · DoS GoldenEye · DoS Hulk · DoS Slowhttptest
DoS Slowloris · FTP-Patator · Heartbleed · Infiltration · PortScan
SSH-Patator · Web Attack Brute Force · Web Attack XSS · Web Attack Sql Injection
```

---

## Estrutura do projeto

```
projeto-ml-ids/
├── src/
│   ├── data/
│   │   └── loader.py          # Ingestão, limpeza, split estratificado
│   ├── features/
│   │   ├── resampling.py      # SMOTE seletivo (RandomOverSampler + SMOTE)
│   │   └── selection.py       # Feature selection via SHAP
│   ├── models/
│   │   └── tuning.py          # Optuna — tuning XGBoost e LightGBM
│   ├── evaluation/
│   │   └── metrics.py         # Métricas, classification report, confusion matrix
│   └── api/                   # FastAPI (fase 3)
├── tests/                     # Espelha src/
├── notebooks/                 # Exploração e prototipação
├── reports/
│   └── figures/               # Confusion matrices e plots de importância
├── artifacts/                 # Modelos serializados (gitignored)
├── data/
│   ├── raw/                   # CSVs originais do CICIDS2017 (gitignored)
│   └── processed/             # Dados limpos (gitignored)
├── train.py                   # Entry point CLI de treino
├── predict.py                 # Entry point CLI de predição
└── pyproject.toml
```

---

## Pipeline

```
CSVs brutos
    │
    ▼
load_csvs()           ← concatena os 8 arquivos do CICIDS2017
    │
    ▼
clean()               ← remove IDs, trata inf/NaN, remove constantes e duplicatas
    │
    ▼
normalize_labels()    ← padroniza nomes das 15 classes
    │
    ▼
encode_labels()       ← LabelEncoder → inteiros
    │
    ▼
train_test_split()    ← estratificado, test_size=0.2, seed=42
    │
    ├──[--resample]──▶ apply_resampling()   ← RandomOverSampler → SMOTE seletivo
    │                                          (somente no treino)
    ├──[--select]────▶ select_features()    ← SHAP TreeExplainer → top-k features
    ├──[--tune]──────▶ tune_xgboost/lgbm() ← Optuna, 50 trials, CV 5-fold, Macro F1
    │
    ▼
fit() + predict()     ← RandomForest / XGBoost / LightGBM
    │
    ▼
MLflow logging        ← params, métricas, modelo, confusion matrix
```

---

## Como rodar

### Pré-requisitos

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -e ".[dev]"
```

Coloque os CSVs do CICIDS2017 em `data/raw/`.

### Treino

```bash
# Baseline — RF + XGBoost + LightGBM
python train.py

# Com SMOTE nas classes minoritárias
python train.py --resample

# Com seleção de features via SHAP (top 40 features)
python train.py --select --n-features 40

# Pipeline completo: resampling + seleção + tuning Optuna
python train.py --resample --select --tune --n-trials 50

# Treinar apenas modelos específicos
python train.py --models XGBoost LightGBM --resample
```

### Experimentos no MLflow

```bash
mlflow ui --backend-store-uri mlruns/
# Abra http://localhost:5000
```

---

## Estratégia de resampling

O dataset CICIDS2017 é severamente desbalanceado — classes como *Heartbleed* e *Infiltration* têm menos de 50 amostras no treino.

A estratégia é aplicada **somente no conjunto de treino**, em dois passos:

1. **RandomOverSampler** — eleva classes com < 50 amostras até 50 (viabiliza SMOTE).
2. **SMOTE** — sobreamostra classes com < 3.000 amostras até o alvo (padrão: 3.000).

`k_neighbors` do SMOTE é ajustado automaticamente para evitar erros em classes muito raras.

---

## Seleção de features (SHAP)

`src/features/selection.py` usa `shap.TreeExplainer` para calcular a importância média absoluta de cada feature em um modelo tree-based já treinado.

```python
from src.features.selection import ShapSelector

selector = ShapSelector(n_features=40)
selector.fit(model, X_train_sample)   # calcula SHAP values
X_train_selected = selector.transform(X_train)
X_test_selected  = selector.transform(X_test)

selector.plot_importance("reports/figures/shap_importance.png")
```

O seletor é serializado junto com o modelo para garantir que `predict.py` use exatamente as mesmas colunas.

---

## Decisões técnicas

| Decisão | Razão |
|---|---|
| Macro F1 como critério | Penaliza falhas em classes raras; accuracy seria enganosa no desbalanceamento |
| SMOTE somente no treino | Evitar data leakage que infla métricas artificialmente |
| Subset de 150k amostras no tuning | Velocidade; dataset completo tornaria 50 trials inviáveis |
| `k_neighbors` adaptativo no SMOTE | Evita `ValueError` em classes com < 6 amostras |
| MLflow com URI absoluta | Contorna problema de encoding de espaços no path do Windows |
| `LabelEncoder` serializado junto com o modelo | Garantia de consistência entre treino e inferência |

---

## Fases do projeto

- [x] **Fase 1** — Ingestão, limpeza e split (`src/data/loader.py`)
- [x] **Fase 1** — Baselines RF + XGBoost — Macro F1: RF=0.86, XGB=0.89
- [x] **Fase 2** — LightGBM + SMOTE seletivo + tuning Optuna
- [x] **Fase 2** — Feature selection com SHAP (`src/features/selection.py`)
- [ ] **Fase 3** — Ensemble final
- [ ] **Fase 3** — API FastAPI + Dockerfile
- [ ] **Fase 3** — Relatório final

---

## Dataset

**CIC-IDS-2017** — Canadian Institute for Cybersecurity  
Tráfego de rede capturado durante 5 dias com 14 tipos de ataque realistas.  
~2.8 milhões de fluxos, 78 features extraídas pelo CICFlowMeter.

> Os arquivos originais não estão incluídos neste repositório. Faça o download em:  
> https://www.unb.ca/cic/datasets/ids-2017.html

---

## Licença

MIT
