# Projeto: Classificador de Tráfego Malicioso com Machine Learning (CICIDS2017)

## Objetivo
Construir um classificador multiclasse de tráfego malicioso treinado no **CICIDS2017**, com foco em detecção prática e robusta de **benigno + 14 categorias de ataque**, priorizando:
- boa generalização;
- baixo falso positivo;
- bom desempenho nas classes raras;
- pipeline reproduzível e fácil de evoluir.

## Tese do projeto
O CICIDS2017 costuma responder muito bem a **modelos baseados em árvore e ensembles**, especialmente **Random Forest** e **XGBoost**, enquanto **stacking/blending** e **seleção de features** frequentemente elevam o teto de performance. Ao mesmo tempo, o dataset tem **desbalanceamento de classes** e exige cuidado com **leakage** e com a forma de divisão dos dados.  
Por isso, o projeto deve ser montado como um sistema de experimentação sério, não como um único treino “mágico”.

## Escopo técnico
### Versão 1 — baseline forte
Implementar um pipeline completo para:
1. carregar e limpar os CSVs do CICIDS2017;
2. normalizar o nome das classes;
3. tratar valores ausentes e infinitos;
4. remover colunas que vazam informação ou identificadores inúteis para o modelo;
5. treinar e comparar modelos clássicos;
6. salvar métricas, artefatos e modelo final.

### Versão 2 — melhoria de performance
Adicionar:
- seleção de features;
- tratamento de desbalanceamento;
- ajuste fino de hiperparâmetros;
- explicabilidade;
- API de inferência.

### Fora do escopo inicial
- deep learning pesado em raw packets;
- captura em tempo real da rede;
- integração com SIEM/EDR;
- detecção adversarial;
- treino federado.

## Estratégia de modelagem
### Modelos obrigatórios
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

### Modelos opcionais para comparação
- LightGBM
- CatBoost
- SVM
- MLP simples

### Ensemble final candidato
- Stacking com metamodelo leve
- Ou votação ponderada entre RF + XGBoost + LightGBM

## Por que esses modelos?
O histórico de resultados no CICIDS2017 favorece modelos que:
- lidam bem com relações não lineares;
- capturam interações entre features de fluxo;
- são resistentes a ruído;
- permitem ranking de importância de variáveis.

Na prática:
- **Random Forest** funciona como baseline forte e interpretável;
- **XGBoost** costuma ser um dos melhores trade-offs entre precisão, recall e custo;
- **stacking** pode elevar a performance final, mas aumenta complexidade e risco de overfitting;
- **feature selection** costuma ser decisiva porque o dataset é grande e nem toda coluna agrega sinal real.

## Pipeline de dados
### Ingestão
- ler os arquivos do CICIDS2017;
- padronizar encoding e tipos;
- consolidar em um dataset único.

### Limpeza
- substituir `inf`, `-inf` e `NaN`;
- remover duplicatas;
- remover colunas de identificação se elas não forem úteis para o caso de uso.

### Split
- separar em treino/validação/teste de forma estratificada;
- preferir divisão que reduza vazamento entre fluxos correlatos, quando possível.

### Pré-processamento
- scaler somente para modelos que precisarem;
- encoding apenas se houver variáveis categóricas remanescentes;
- pipeline reproduzível com `Pipeline` do scikit-learn.

## Tratamento de desbalanceamento
Testar, nesta ordem:
1. `class_weight`;
2. undersampling controlado;
3. SMOTE;
4. SMOTE-Tomek ou ADASYN, se fizer sentido nos experimentos.

Regra: qualquer técnica de reamostragem deve ser aplicada **apenas no treino**.

## Seleção de features
Usar pelo menos duas abordagens:
- importâncias do Random Forest / XGBoost;
- seleção por informação mútua ou correlação;
- opcional: SHAP para validar as features mais relevantes.

Meta: reduzir dimensionalidade sem matar recall das classes raras.

## Métricas de sucesso
Não aceitar só accuracy.

### Métricas principais
- Accuracy
- Macro F1
- Weighted F1
- Recall por classe
- Precision por classe
- Matriz de confusão
- ROC-AUC one-vs-rest, se aplicável

### Critérios de aceite
- manter **96%+ de accuracy** como linha de base;
- elevar o **Macro F1** acima do baseline;
- evitar colapso nas classes minoritárias;
- reduzir falso positivo em benigno.

## Skills que este projeto exige
### Hard skills
- Python
- Pandas / NumPy
- scikit-learn
- Classificação multiclasse
- Feature engineering
- Tratamento de desbalanceamento
- Validação experimental
- Análise de métricas
- Serialização de modelos

### Skills avançadas
- XGBoost / ensemble methods
- explicabilidade com SHAP
- desenho de pipeline limpo
- engenharia de experimentos
- leitura crítica de benchmarks
- noções de detecção de intrusão e tráfego em rede

## Tecnologias recomendadas
### Core
- Python 3.11+
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- xgboost
- lightgbm
- joblib

### Experimentação
- optuna
- matplotlib
- seaborn
- shap
- mlflow ou Weights & Biases, se quiser rastrear experimentos

### Engenharia
- pytest
- ruff
- black
- mypy
- Git
- Docker

### Entrega
- FastAPI para servir inferência
- pydantic para schema de entrada
- uvicorn para execução local

## Estrutura sugerida do repositório
```text
.
├── data/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── api/
├── tests/
├── artifacts/
├── reports/
├── train.py
├── predict.py
├── app.py
├── requirements.txt
└── README.md
```

## Plano de execução
### Fase 1 — base
- explorar o dataset;
- limpar e unificar labels;
- treinar Logistic Regression, Decision Tree e Random Forest;
- registrar baseline.

### Fase 2 — otimização
- testar XGBoost, LightGBM e CatBoost;
- aplicar feature selection;
- testar reamostragem;
- ajustar hiperparâmetros.

### Fase 3 — produto
- selecionar o melhor modelo;
- exportar artefato;
- criar API de predição;
- documentar entradas, saídas e limitações.

## Entregáveis
- notebook de exploração;
- pipeline de treino reproduzível;
- tabela comparativa de modelos;
- modelo final serializado;
- API local de inferência;
- relatório curto com conclusões técnicas.

## Prompt de trabalho para o Claude Code
Implementar um classificador multiclasse para CICIDS2017 com foco em robustez e clareza. Começar com limpeza dos dados, split seguro, treino de baselines e avaliação por Macro F1 e recall por classe. Depois testar Random Forest, XGBoost e um ensemble simples, com seleção de features e tratamento de desbalanceamento apenas no treino. Finalizar com exportação do melhor modelo, documentação e uma API FastAPI para inferência.

## Diretriz final
Este projeto deve ser tratado como um estudo de **detecção de intrusão aplicada**, não como um exercício de “ganhar accuracy”.  
A meta real é: **ser forte no benchmark, mas também confiável, reproduzível e utilizável**.
