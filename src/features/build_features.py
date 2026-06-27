"""
Módulo de engenharia de features.

Constrói o pipeline de pré-processamento compatível com sklearn:
- Codificação ordinal/one-hot para categóricas
- Escalonamento padrão para numéricas
- Features de domínio específico
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Engenharia de features de domínio
# ------------------------------------------------------------------

class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Cria features de domínio antes da codificação.

    Novas features
    --------------
    - gasto_mensal_medio    : TotalCharges / (tenure + 1)
    - gasto_por_servico     : MonthlyCharges / (n_servicos + 1)
    - n_servicos            : contagem de serviços adicionais ativos
    - contrato_longo_prazo  : 1 se Contrato != 'Month-to-month'
    - grupo_tenure          : tenure agrupado (0-12, 13-24, 25-48, 49-72)
    - tendencia_cobranca    : MonthlyCharges vs gasto_mensal_medio
    """

    SERVICOS_ADICIONAIS = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Número de serviços adicionais ativos
        flags_servicos = df[self.SERVICOS_ADICIONAIS].apply(
            lambda col: (col == "Yes").astype(int)
        )
        df["n_servicos"] = flags_servicos.sum(axis=1)

        # Features de gasto
        df["gasto_mensal_medio"] = df["TotalCharges"] / (df["tenure"] + 1)
        df["gasto_por_servico"] = df["MonthlyCharges"] / (df["n_servicos"] + 1)
        df["tendencia_cobranca"] = df["MonthlyCharges"] - df["gasto_mensal_medio"]

        # Flag de tipo de contrato
        df["contrato_longo_prazo"] = (
            df["Contract"].isin(["One year", "Two year"])
        ).astype(int)

        # Grupos de tenure
        df["grupo_tenure"] = pd.cut(
            df["tenure"],
            bins=[0, 12, 24, 48, 72],
            labels=["0-12m", "13-24m", "25-48m", "49-72m"],
            include_lowest=True,
        ).astype(str)

        logger.debug(f"Features de domínio adicionadas. Shape: {df.shape}")
        return df


# ------------------------------------------------------------------
# Fábrica do pipeline de pré-processamento
# ------------------------------------------------------------------

FEATURES_CATEGORICAS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "grupo_tenure",
]

FEATURES_NUMERICAS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "n_servicos",
    "gasto_mensal_medio",
    "gasto_por_servico",
    "tendencia_cobranca",
    "contrato_longo_prazo",
]

# Aliases em inglês para compatibilidade
CATEGORICAL_FEATURES = FEATURES_CATEGORICAS
NUMERICAL_FEATURES = FEATURES_NUMERICAS


def construir_preprocessador(
    features_categoricas: Optional[List[str]] = None,
    features_numericas: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Constrói o ColumnTransformer do sklearn para pré-processamento.

    Parâmetros
    ----------
    features_categoricas : lista de str, opcional
        Nomes das colunas categóricas. Padrão: FEATURES_CATEGORICAS.
    features_numericas : lista de str, opcional
        Nomes das colunas numéricas. Padrão: FEATURES_NUMERICAS.

    Retorna
    -------
    ColumnTransformer
    """
    cat_feats = features_categoricas or FEATURES_CATEGORICAS
    num_feats = features_numericas or FEATURES_NUMERICAS

    pipeline_categorico = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"),
            )
        ]
    )

    pipeline_numerico = Pipeline(
        steps=[("escalonador", StandardScaler())]
    )

    preprocessador = ColumnTransformer(
        transformers=[
            ("num", pipeline_numerico, num_feats),
            ("cat", pipeline_categorico, cat_feats),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessador


def construir_pipeline_completo(modelo) -> Pipeline:
    """
    Encadeia engenharia de features + pré-processamento + modelo em um único Pipeline.

    Parâmetros
    ----------
    modelo : estimador sklearn
        Qualquer modelo compatível com sklearn (ajustado ou não).

    Retorna
    -------
    sklearn.pipeline.Pipeline
    """
    pipeline = Pipeline(
        steps=[
            ("engenharia_features", ChurnFeatureEngineer()),
            ("preprocessador", construir_preprocessador()),
            ("classificador", modelo),
        ]
    )
    return pipeline


def obter_nomes_features(pipeline: Pipeline) -> List[str]:
    """Extrai os nomes das features após o pré-processamento para análise SHAP."""
    preprocessador = pipeline.named_steps["preprocessador"]
    return list(preprocessador.get_feature_names_out())


# Aliases em inglês para compatibilidade com notebooks
build_preprocessor = construir_preprocessador
build_full_pipeline = construir_pipeline_completo
get_feature_names = obter_nomes_features
