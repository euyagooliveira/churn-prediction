"""
Módulo de carregamento e pré-processamento de dados.

Responsável por toda a ingestão, limpeza e conversão de tipos
antes da engenharia de features.
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def carregar_dados_brutos(caminho: str | Path) -> pd.DataFrame:
    """
    Carrega o dataset Telco Customer Churn a partir de um CSV.

    Parâmetros
    ----------
    caminho : str ou Path
        Caminho para o arquivo CSV bruto.

    Retorna
    -------
    pd.DataFrame
        DataFrame bruto com o schema original.
    """
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em {caminho}. "
            "Baixe em: https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
        )
    df = pd.read_csv(caminho)
    logger.info(f"Carregados {len(df):,} registros de {caminho}")
    return df


def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica as etapas de limpeza ao DataFrame bruto:
    - Corrige o dtype de TotalCharges (carregado como object por strings vazias)
    - Preenche TotalCharges ausente com MonthlyCharges (novos clientes, tenure=0)
    - Codifica a variável alvo como inteiro
    - Remove espaços em branco das colunas de texto

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame bruto.

    Retorna
    -------
    pd.DataFrame
        DataFrame limpo.
    """
    df = df.copy()

    # Corrige TotalCharges: converte strings vazias para NaN e preenche
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    total_ausentes = df["TotalCharges"].isna().sum()
    if total_ausentes > 0:
        logger.info(
            f"Preenchendo {total_ausentes} valores ausentes em TotalCharges com MonthlyCharges"
        )
        df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])

    # Codifica a variável alvo
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    taxa_churn = df["Churn"].mean()
    logger.info(f"Taxa de churn: {taxa_churn:.2%} ({df['Churn'].sum():,} churners)")

    # Remove espaços em branco de todas as colunas de texto
    colunas_texto = df.select_dtypes("object").columns
    df[colunas_texto] = df[colunas_texto].apply(lambda col: col.str.strip())

    return df


def dividir_dados(
    df: pd.DataFrame,
    alvo: str = "Churn",
    tamanho_teste: float = 0.2,
    semente_aleatoria: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divisão estratificada treino/teste preservando a distribuição das classes.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame limpo.
    alvo : str
        Nome da coluna alvo.
    tamanho_teste : float
        Proporção do conjunto de teste.
    semente_aleatoria : int
        Semente para reprodutibilidade.

    Retorna
    -------
    X_treino, X_teste, y_treino, y_teste
    """
    X = df.drop(columns=[alvo, "customerID"], errors="ignore")
    y = df[alvo]

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y,
        test_size=tamanho_teste,
        stratify=y,
        random_state=semente_aleatoria,
    )

    logger.info(
        f"Treino: {len(X_treino):,} registros | Teste: {len(X_teste):,} registros | "
        f"Taxa de churn no treino: {y_treino.mean():.2%}"
    )
    return X_treino, X_teste, y_treino, y_teste


def resumo_dados(df: pd.DataFrame) -> dict:
    """
    Retorna um resumo conciso do DataFrame para relatórios.
    """
    return {
        "n_linhas": len(df),
        "n_colunas": len(df.columns),
        "valores_ausentes": df.isnull().sum().to_dict(),
        "tipos": df.dtypes.astype(str).to_dict(),
        "taxa_churn": df["Churn"].mean() if "Churn" in df.columns else None,
        "estatisticas_numericas": df.select_dtypes("number").describe().to_dict(),
    }


# Aliases em inglês para compatibilidade com notebooks antigos
load_raw_data = carregar_dados_brutos
clean_data = limpar_dados
split_data = dividir_dados
get_data_summary = resumo_dados
