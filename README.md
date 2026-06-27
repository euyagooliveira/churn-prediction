# Previsão de Churn de Clientes

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange?logo=scikit-learn)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Pipeline de machine learning end-to-end para prever cancelamento de clientes em uma empresa de telecomunicações. Construído com código de qualidade de produção: arquitetura modular em `src/`, pipelines sklearn, explicabilidade com SHAP e cobertura completa de testes unitários.

---

## Problema

Uma empresa de telecom quer identificar clientes com risco de churn (cancelamento) antes que eles saiam, permitindo ações de retenção direcionadas. O custo de adquirir um novo cliente é 5–25× maior do que reter um existente.

**Objetivo de negócio:** Construir um modelo que maximize o recall sobre churners mantendo precisão aceitável — capturando o maior número possível de clientes em risco.

---

## Resultados

| Métrica | Valor |
|---------|-------|
| **ROC-AUC** | **0.858** |
| **Average Precision** | **0.682** |
| **F1 Score** (threshold otimizado) | **0.622** |
| **Recall** (detecção de churners) | **0.80** |
| **Precisão** | **0.51** |
| **CV ROC-AUC (5 folds)** | **0.856 ± 0.008** |

> Threshold de decisão otimizado para **0.40** (vs padrão 0.50), melhorando o F1 em ~8%.

---

## Estrutura do Projeto

```
churn-prediction/
├── data/
│   ├── raw/                    # CSV bruto (não versionado — ver seção Dados)
│   └── processed/              # Dados intermediários limpos
│
├── notebooks/
│   ├── 01_eda.ipynb            # Análise Exploratória de Dados
│   ├── 02_feature_engineering.ipynb  # Criação e validação de features
│   └── 03_modeling.ipynb       # Treinamento, avaliação e SHAP
│
├── src/
│   ├── data/
│   │   └── preprocessing.py    # Carregamento, limpeza e divisão dos dados
│   ├── features/
│   │   └── build_features.py   # Engenharia de features + pipeline sklearn
│   ├── models/
│   │   ├── train.py            # Treinamento, CV e persistência do modelo
│   │   └── evaluate.py         # Métricas, análise de threshold e de erros
│   └── visualization/
│       └── plots.py            # Funções de visualização reutilizáveis
│
├── tests/
│   ├── test_preprocessing.py   # Testes unitários do módulo de dados
│   └── test_features.py        # Testes unitários do módulo de features
│
├── reports/figures/            # Gráficos gerados automaticamente
├── config.yaml                 # Configuração centralizada
├── requirements.txt
└── setup.py
```

---

## Início Rápido

### 1. Clonar e instalar

```bash
git clone https://github.com/yourusername/churn-prediction.git
cd churn-prediction
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 2. Obter os dados

Baixe o dataset [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) do Kaggle e coloque o CSV em:

```
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### 3. Executar os notebooks (recomendado)

```bash
jupyter notebook notebooks/
```

Execute na ordem: `01_eda` → `02_feature_engineering` → `03_modeling`

### 4. Treinar via CLI

```bash
python src/models/train.py --config config.yaml
```

### 5. Executar os testes

```bash
pytest tests/ -v --cov=src
```

---

## Principais Descobertas

**Principais drivers de churn (análise SHAP):**

1. **Tipo de contrato** — clientes mensais cancelam a 42% vs 3% em contratos de 2 anos
2. **Tempo de contrato (tenure)** — primeiros 12 meses apresentam risco ~3× maior
3. **Fibra óptica** — taxa de churn mais alta (41%) em relação a DSL ou sem internet
4. **Sem TechSupport / OnlineSecurity** — forte sinal de churn
5. **Pagamento por cheque eletrônico** — maior taxa de churn entre todos os métodos

**Recomendação de negócio:** concentrar o orçamento de retenção em clientes mensais no primeiro ano com serviço de fibra óptica e sem pacotes de suporte.

---

## Abordagem de Modelagem

### Engenharia de Features

Além das 20 features originais, foram criadas 6 features de domínio:

| Feature | Lógica | Intuição de Negócio |
|---------|--------|---------------------|
| `n_servicos` | Contagem de add-ons ativos | Mais serviços = maior custo de troca |
| `gasto_mensal_medio` | TotalCharges / (tenure+1) | Taxa histórica de gasto |
| `gasto_por_servico` | MonthlyCharges / (n_servicos+1) | Eficiência de gasto |
| `tendencia_cobranca` | MonthlyCharges - gasto_mensal_medio | As cobranças estão aumentando? |
| `contrato_longo_prazo` | Contrato != mensal | Flag de comprometimento |
| `grupo_tenure` | Tenure em faixas (0-12, 13-24...) | Efeitos não lineares do tenure |

### Comparação de Modelos

4 modelos foram comparados com validação cruzada estratificada de 5 folds:

| Modelo | CV ROC-AUC |
|--------|------------|
| Regressão Logística | 0.840 ± 0.009 |
| Random Forest | 0.838 ± 0.011 |
| **XGBoost** | **0.856 ± 0.008** |
| LightGBM | 0.852 ± 0.010 |

**XGBoost** selecionado como melhor modelo. Desbalanceamento de classes tratado via `scale_pos_weight=2.7`.

### Arquitetura do Pipeline

```
Dados Brutos
    |
    v
ChurnFeatureEngineer     <- Transformer sklearn customizado
    |  (adiciona 6 features)
    v
ColumnTransformer
    |-- StandardScaler   <- features numéricas
    +-- OneHotEncoder    <- features categóricas
    |
    v
XGBClassifier
    |
    v
Predição + Ajuste de Threshold (0.40)
```

---

## Testes

```bash
pytest tests/ -v
```

```
tests/test_preprocessing.py::TestLimpezaDados::test_totalcharges_convertido_para_float PASSED
tests/test_preprocessing.py::TestLimpezaDados::test_totalcharges_ausente_preenchido PASSED
tests/test_preprocessing.py::TestLimpezaDados::test_churn_codificado_como_int PASSED
tests/test_preprocessing.py::TestDivisaoDados::test_tamanhos_corretos PASSED
tests/test_preprocessing.py::TestDivisaoDados::test_target_fora_das_features PASSED
tests/test_features.py::TestChurnFeatureEngineer::test_adiciona_n_servicos PASSED
tests/test_features.py::TestChurnFeatureEngineer::test_contrato_longo_prazo PASSED
tests/test_features.py::TestPipelineCompleto::test_pipeline_treina_e_prediz PASSED
... (16 testes no total)
```

---

## Configuracao

Todos os parâmetros estão centralizados em `config.yaml`:

```yaml
modelo:
  algoritmo: "xgboost"
  hiperparametros:
    xgboost:
      n_estimators: 300
      learning_rate: 0.05
      scale_pos_weight: 2.7  # trata desbalanceamento de classes

avaliacao:
  threshold: 0.40  # ajustado para recall
  cv_folds: 5
```

---

## Stack

- **Python 3.10+**
- **pandas / numpy** — manipulação de dados
- **scikit-learn** — pipelines, pré-processamento e avaliação
- **XGBoost / LightGBM** — gradient boosting
- **imbalanced-learn** — tratamento de desbalanceamento de classes
- **SHAP** — explicabilidade do modelo
- **matplotlib / seaborn** — visualização
- **pytest** — testes unitários

---

## Próximos Passos

- [ ] Otimização de hiperparâmetros com Optuna
- [ ] Rastreamento de experimentos com MLflow
- [ ] Endpoint de inferência com FastAPI
- [ ] Containerização com Docker
- [ ] Integração com feature store (Feast)

---

## Licença

MIT License — veja [LICENSE](LICENSE) para detalhes.

---

*Desenvolvido por [Yago Oliveira](https://github.com/yourusername)*
