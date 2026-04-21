# TODO — Melhorias Futuras (Macro F1 atual: 0.89 → meta: >0.93)

> Propostas priorizadas por impacto estimado no Macro F1 e esforço de implementação.  
> Classes-alvo: **Web Attack XSS (F1=0.39)**, **Bot (F1=0.73)**, **Web Attack Brute Force (F1=0.71)**.

---

## 🔴 Alta prioridade — maior impacto esperado

### 1. Classificador hierárquico em dois níveis

**Problema:** O modelo trata Benign vs. todos os ataques num único espaço de 15 classes — classes com comportamento semelhante (ex.: Web Attack XSS vs. Brute Force) se confundem.

**Proposta:**
- **Nível 1:** classificador binário `Benign | Attack` (recall altíssimo, baixo FP)
- **Nível 2:** classificador multiclasse apenas sobre os fluxos classificados como `Attack`

**Benefício:** O nível 2 treina num espaço reduzido (14 classes) sem o domínio massivo de Benign distorcendo as fronteiras.

**Arquivo:** `src/models/hierarchical.py`  
**Esforço:** Médio | **Impacto estimado:** +0.03–0.05 Macro F1

---

### 2. Engenharia de features específicas para Web Attacks

**Problema:** XSS e Brute Force têm F1 baixo porque as features de fluxo (bytes, pacotes, duração) não capturam o conteúdo do payload HTTP.

**Proposta:**
- Extrair features de **payload HTTP** via CICFlowMeter com flags habilitadas ou **nfstream** com DPI
- Features candidatas: tamanho médio de URL, entropia do payload, contagem de parâmetros GET/POST, presença de caracteres especiais (`<`, `>`, `'`, `"`, `%27`)
- Para Bot: adicionar features de **periodicidade temporal** (desvio padrão do inter-arrival time por IP de origem)

**Arquivo:** `src/features/http_features.py`  
**Esforço:** Alto (requer reprocessar PCAPs) | **Impacto estimado:** +0.04–0.08 em XSS

---

### 3. CTGAN para geração sintética das classes raras

**Problema:** SMOTE gera interpolação linear — ruim para classes com distribuição multimodal (Bot, Web Attacks).

**Proposta:**
- Usar **CTGAN** (Conditional Tabular GAN) ou **TVAE** da biblioteca `sdv` para gerar amostras sintéticas mais realistas para as 5 classes mais fracas
- Comparar via Macro F1 em CV contra SMOTE atual

```bash
pip install sdv
```

**Arquivo:** `src/features/synthetic.py`  
**Esforço:** Médio | **Impacto estimado:** +0.02–0.04 Macro F1

---

### 4. Threshold customizado por classe (calibração de probabilidades)

**Problema:** O modelo usa threshold padrão de 0.5 para todas as classes — sub-ótimo para classes com suporte pequeno.

**Proposta:**
- Calibrar probabilidades com **Platt Scaling** (`CalibratedClassifierCV`) ou **Isotonic Regression**
- Otimizar threshold por classe via busca em grid no conjunto de validação, maximizando F1 por classe
- Priorizar recall em Infiltration/Heartbleed e precisão em Benign

**Arquivo:** `src/models/calibration.py`  
**Esforço:** Baixo | **Impacto estimado:** +0.02–0.03 Macro F1

---

## 🟡 Média prioridade — impacto moderado

### 5. Tuning Optuna com foco em Macro F1 ponderado por classe rara

**Problema:** O tuning atual otimiza Macro F1 médio — as 3 classes problemáticas têm peso igual às 12 outras.

**Proposta:**
- Criar métrica customizada: `weighted_macro_f1` com peso 3× para XSS, Bot, Brute Force
- Usar essa métrica como objetivo no Optuna

**Arquivo:** `src/models/tuning.py` (extensão)  
**Esforço:** Baixo | **Impacto estimado:** +0.01–0.02 nas classes-alvo

---

### 6. Modelo especialista para Web Attacks (One-vs-Rest fine-tuned)

**Proposta:**
- Treinar um classificador binário dedicado `Web Attack XSS | outros`
- Usar suas probabilidades como feature adicional no meta-learner do Stacking
- Testar também com **TabNet** (rede neural tabular) que tende a capturar interações não-lineares que GBM perde

**Arquivo:** `src/models/specialist.py`  
**Esforço:** Médio | **Impacto estimado:** +0.02–0.04 em XSS

---

### 7. Validação temporal (cross-day split)

**Problema:** O split aleatório atual vaza padrões temporais — em produção, o modelo vê tráfego de dias futuros.

**Proposta:**
- Treinar nos dias Monday–Thursday, testar em Friday
- Mede degradação real de generalização e identifica features que overfitam ao dia

**Arquivo:** `src/data/loader.py` (nova opção `--split-by-day`)  
**Esforço:** Baixo | **Impacto:** Métrica mais realista, pode revelar overfitting oculto

---

### 8. Feature selection por grupo de classe (SHAP por subconjunto)

**Problema:** As top-40 features por SHAP global podem não ser as melhores para discriminar XSS vs. outros.

**Proposta:**
- Calcular SHAP values separadamente para cada classe problemática
- Criar feature sets específicos para o nível 2 do classificador hierárquico

**Arquivo:** `src/features/selection.py` (método `fit_per_class`)  
**Esforço:** Médio | **Impacto estimado:** +0.01–0.03 nas classes-alvo

---

## 🟢 Baixa prioridade — melhorias de infraestrutura

### 9. Monitoramento de drift em produção

**Proposta:**
- Implementar detector de **data drift** com `evidently` ou `nannyml`
- Alertar quando a distribuição de features de entrada diverge do treino (útil para integração Wazuh)
- Dashboard de drift integrado ao MLflow ou Grafana

**Arquivo:** `src/monitoring/drift.py`  
**Esforço:** Médio | **Impacto:** Confiabilidade em produção

---

### 10. Autenticação na API + rate limiting

**Proposta:**
- Adicionar Bearer token (FastAPI + `python-jose`)
- Rate limiting por IP com `slowapi`
- Necessário antes de qualquer deploy externo (Wazuh, SIEM)

**Arquivo:** `src/api/app.py` (extensão)  
**Esforço:** Baixo | **Impacto:** Segurança em produção

---

### 11. Pipeline CI/CD com GitHub Actions

**Proposta:**
- Rodar `pytest` a cada push
- Verificar Macro F1 mínimo (0.85) com dados sintéticos no CI
- Build e push automático da imagem Docker no merge para `main`

**Arquivo:** `.github/workflows/ci.yml`  
**Esforço:** Baixo | **Impacto:** Qualidade e reprodutibilidade

---

## Ordem de execução sugerida

```
1. Calibração de threshold      (baixo esforço, ganho imediato)
2. Tuning com métrica ponderada (baixo esforço, foco nas classes fracas)
3. CTGAN para classes raras     (médio esforço, alternativa ao SMOTE)
4. Classificador hierárquico    (médio esforço, maior ganho arquitetural)
5. Features HTTP (payload)      (alto esforço, necessita reprocessar PCAPs)
```

---

## Referências

- **CTGAN:** Xu et al. (2019). *Modeling Tabular Data using Conditional GAN*. NeurIPS.
- **TabNet:** Arik & Pfister (2021). *TabNet: Attentive Interpretable Tabular Learning*. AAAI.
- **Calibração:** Niculescu-Mizil & Caruana (2005). *Predicting Good Probabilities With Supervised Learning*. ICML.
- **Hierarchical Classification:** Silla & Freitas (2011). *A survey of hierarchical classification across different application domains*. DMKD.
- **Evidently AI:** https://github.com/evidentlyai/evidently
