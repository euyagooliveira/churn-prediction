"""
Módulo de avaliação de modelos.

Fornece métricas abrangentes, análise de threshold e
utilitários de análise de erros.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def calcular_metricas(
    y_verdadeiro: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calcula um conjunto abrangente de métricas de classificação binária.

    Parâmetros
    ----------
    y_verdadeiro : array-like
        Rótulos verdadeiros.
    y_pred : array-like
        Predições binárias (já com threshold aplicado).
    y_prob : array-like
        Probabilidades preditas para a classe positiva.
    threshold : float
        Threshold de decisão utilizado (apenas para registro).

    Retorna
    -------
    dict de nome_metrica -> valor
    """
    metricas = {
        "roc_auc": roc_auc_score(y_verdadeiro, y_prob),
        "average_precision": average_precision_score(y_verdadeiro, y_prob),
        "f1": f1_score(y_verdadeiro, y_pred),
        "precisao": precision_score(y_verdadeiro, y_pred),
        "recall": recall_score(y_verdadeiro, y_pred),
        "threshold": threshold,
    }

    logger.info("=" * 50)
    logger.info("Resultados da Avaliação")
    logger.info("=" * 50)
    for k, v in metricas.items():
        logger.info(f"  {k:>20}: {v:.4f}")
    logger.info(
        "\n" + classification_report(
            y_verdadeiro, y_pred, target_names=["Sem Churn", "Churn"]
        )
    )

    return metricas


def encontrar_threshold_otimo(
    y_verdadeiro: np.ndarray,
    y_prob: np.ndarray,
    metrica: str = "f1",
) -> Tuple[float, float]:
    """
    Encontra o threshold de probabilidade que maximiza uma métrica.

    Parâmetros
    ----------
    y_verdadeiro : array-like
        Rótulos verdadeiros.
    y_prob : array-like
        Probabilidades preditas.
    metrica : str
        'f1', 'recall' ou 'precisao'.

    Retorna
    -------
    (threshold_otimo, melhor_score)
    """
    precisoes, recalls, thresholds = precision_recall_curve(y_verdadeiro, y_prob)

    if metrica == "f1":
        # Evita divisão por zero
        scores_f1 = np.where(
            (precisoes + recalls) > 0,
            2 * (precisoes * recalls) / (precisoes + recalls),
            0,
        )
        melhor_idx = np.argmax(scores_f1[:-1])
        melhor_score = scores_f1[melhor_idx]
    elif metrica == "recall":
        melhor_idx = np.argmax(recalls[:-1])
        melhor_score = recalls[melhor_idx]
    elif metrica in ("precisao", "precision"):
        melhor_idx = np.argmax(precisoes[:-1])
        melhor_score = precisoes[melhor_idx]
    else:
        raise ValueError(
            f"metrica deve ser 'f1', 'recall' ou 'precisao', recebido '{metrica}'"
        )

    threshold_otimo = thresholds[melhor_idx]
    logger.info(
        f"Threshold ótimo para {metrica}: {threshold_otimo:.4f} "
        f"(score: {melhor_score:.4f})"
    )
    return float(threshold_otimo), float(melhor_score)


def avaliar_pipeline(
    pipeline: Pipeline,
    X_teste: pd.DataFrame,
    y_teste: pd.Series,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Avaliação end-to-end de um pipeline ajustado nos dados de teste.

    Parâmetros
    ----------
    pipeline : Pipeline
        Pipeline sklearn ajustado.
    X_teste : pd.DataFrame
        Features de teste (brutas, antes da engenharia de features).
    y_teste : pd.Series
        Alvo de teste.
    threshold : float
        Threshold de decisão para predições binárias.

    Retorna
    -------
    dict de métricas
    """
    y_prob = pipeline.predict_proba(X_teste)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metricas = calcular_metricas(y_teste, y_pred, y_prob, threshold)
    return metricas


def obter_dados_curva_roc(
    y_verdadeiro: np.ndarray, y_prob: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Retorna arrays FPR, TPR e AUC para plotagem."""
    fpr, tpr, _ = roc_curve(y_verdadeiro, y_prob)
    auc = roc_auc_score(y_verdadeiro, y_prob)
    return fpr, tpr, auc


def obter_matriz_confusao(
    y_verdadeiro: np.ndarray, y_pred: np.ndarray
) -> np.ndarray:
    """Retorna a matriz de confusão como array numpy."""
    return confusion_matrix(y_verdadeiro, y_pred)


def analisar_erros(
    X_teste: pd.DataFrame,
    y_teste: pd.Series,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, pd.DataFrame]:
    """
    Analisa falsos positivos e falsos negativos.

    Retorna
    -------
    dict com chaves 'falsos_positivos', 'falsos_negativos', 'verdadeiros_positivos', 'verdadeiros_negativos'
    """
    y_pred = (y_prob >= threshold).astype(int)

    df = X_teste.copy()
    df["y_verdadeiro"] = y_teste.values
    df["y_pred"] = y_pred
    df["y_prob"] = y_prob

    return {
        "falsos_positivos": df[(df["y_verdadeiro"] == 0) & (df["y_pred"] == 1)],
        "falsos_negativos": df[(df["y_verdadeiro"] == 1) & (df["y_pred"] == 0)],
        "verdadeiros_positivos": df[(df["y_verdadeiro"] == 1) & (df["y_pred"] == 1)],
        "verdadeiros_negativos": df[(df["y_verdadeiro"] == 0) & (df["y_pred"] == 0)],
    }


# Aliases em inglês para compatibilidade
compute_metrics = calcular_metricas
find_optimal_threshold = encontrar_threshold_otimo
evaluate_pipeline = avaliar_pipeline
get_roc_curve_data = obter_dados_curva_roc
get_confusion_matrix = obter_matriz_confusao
error_analysis = analisar_erros
