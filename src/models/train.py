"""
Módulo de treinamento de modelos.

Suporta múltiplos algoritmos com validação cruzada, ajuste de
hiperparâmetros via RandomizedSearchCV e persistência do modelo.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from features.build_features import construir_pipeline_completo

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Registro de modelos disponíveis
# ------------------------------------------------------------------

REGISTRO_MODELOS: Dict[str, Any] = {
    "regressao_logistica": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "xgboost": XGBClassifier,
    "lightgbm": LGBMClassifier,
}


def obter_modelo(algoritmo: str, hiperparametros: Dict) -> Any:
    """
    Instancia um modelo a partir do registro.

    Parâmetros
    ----------
    algoritmo : str
        Um de 'regressao_logistica', 'random_forest', 'xgboost', 'lightgbm'.
    hiperparametros : dict
        Hiperparâmetros para o algoritmo escolhido.

    Retorna
    -------
    Estimador compatível com sklearn
    """
    if algoritmo not in REGISTRO_MODELOS:
        raise ValueError(
            f"Algoritmo '{algoritmo}' desconhecido. "
            f"Escolha entre: {list(REGISTRO_MODELOS.keys())}"
        )
    cls = REGISTRO_MODELOS[algoritmo]
    modelo = cls(**hiperparametros)
    logger.info(f"Instanciado {cls.__name__} com parâmetros: {hiperparametros}")
    return modelo


def treinar_modelo(
    X_treino: pd.DataFrame,
    y_treino: pd.Series,
    algoritmo: str = "xgboost",
    hiperparametros: Optional[Dict] = None,
) -> Pipeline:
    """
    Treina o pipeline completo (engenharia de features + pré-processamento + modelo).

    Parâmetros
    ----------
    X_treino : pd.DataFrame
        Features de treino (brutas, antes da engenharia de features).
    y_treino : pd.Series
        Alvo de treino.
    algoritmo : str
        Nome do algoritmo.
    hiperparametros : dict, opcional
        Hiperparâmetros do modelo. Se None, usa valores padrão.

    Retorna
    -------
    Pipeline
        Pipeline sklearn ajustado.
    """
    if hiperparametros is None:
        hiperparametros = {}

    modelo = obter_modelo(algoritmo, hiperparametros)
    pipeline = construir_pipeline_completo(modelo)

    logger.info(f"Treinando {algoritmo} em {len(X_treino):,} amostras...")
    pipeline.fit(X_treino, y_treino)
    logger.info("Treinamento concluído.")

    return pipeline


def validacao_cruzada(
    X: pd.DataFrame,
    y: pd.Series,
    algoritmo: str = "xgboost",
    hiperparametros: Optional[Dict] = None,
    n_folds: int = 5,
    metrica: str = "roc_auc",
) -> Dict[str, Any]:
    """
    Avalia o pipeline com k-fold estratificado e retorna média ± desvio padrão.

    Parâmetros
    ----------
    X : pd.DataFrame
        Matriz de features (brutas).
    y : pd.Series
        Vetor alvo.
    algoritmo : str
        Nome do algoritmo.
    hiperparametros : dict, opcional
        Hiperparâmetros do modelo.
    n_folds : int
        Número de folds.
    metrica : str
        Métrica de avaliação para cross_val_score.

    Retorna
    -------
    dict com 'media', 'desvio_padrao', 'scores_por_fold'
    """
    from sklearn.model_selection import cross_val_score

    if hiperparametros is None:
        hiperparametros = {}

    modelo = obter_modelo(algoritmo, hiperparametros)
    pipeline = construir_pipeline_completo(modelo)

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=metrica, n_jobs=-1)

    resultado = {
        "media": scores.mean(),
        "desvio_padrao": scores.std(),
        "scores_por_fold": scores.tolist(),
        "metrica": metrica,
        "n_folds": n_folds,
        # Aliases em inglês
        "mean_score": scores.mean(),
        "std_score": scores.std(),
        "fold_scores": scores.tolist(),
    }
    logger.info(
        f"CV {metrica}: {scores.mean():.4f} ± {scores.std():.4f} "
        f"(folds: {[f'{s:.4f}' for s in scores]})"
    )
    return resultado


def salvar_modelo(pipeline: Pipeline, caminho_saida: str | Path) -> None:
    """Persiste o pipeline em disco usando pickle."""
    caminho_saida = Path(caminho_saida)
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho_saida, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info(f"Modelo salvo em {caminho_saida}")


def carregar_modelo(caminho_modelo: str | Path) -> Pipeline:
    """Carrega um pipeline persistido do disco."""
    caminho_modelo = Path(caminho_modelo)
    with open(caminho_modelo, "rb") as f:
        pipeline = pickle.load(f)
    logger.info(f"Modelo carregado de {caminho_modelo}")
    return pipeline


# Aliases em inglês para compatibilidade
train_model = treinar_modelo
cross_validate_pipeline = validacao_cruzada
save_model = salvar_modelo
load_model = carregar_modelo


def main():
    """Ponto de entrada CLI para treinamento."""
    import click

    @click.command()
    @click.option("--config", default="config.yaml", help="Caminho para config.yaml")
    def executar(config: str):
        logging.basicConfig(level=logging.INFO)

        with open(config) as f:
            cfg = yaml.safe_load(f)

        from data.preprocessing import carregar_dados_brutos, limpar_dados, dividir_dados

        df = carregar_dados_brutos(cfg["data"]["raw_path"])
        df = limpar_dados(df)
        X_treino, X_teste, y_treino, y_teste = dividir_dados(
            df,
            tamanho_teste=cfg["data"]["test_size"],
            semente_aleatoria=cfg["data"]["random_state"],
        )

        algo = cfg["model"]["algorithm"]
        params = cfg["model"]["hyperparameters"].get(algo, {})

        # Validação cruzada
        resultado_cv = validacao_cruzada(
            X_treino, y_treino,
            algoritmo=algo,
            hiperparametros=params,
            n_folds=cfg["evaluation"]["cv_folds"],
        )

        # Treino final no conjunto de treino completo
        pipeline = treinar_modelo(X_treino, y_treino, algoritmo=algo, hiperparametros=params)
        salvar_modelo(pipeline, "models/churn_pipeline.pkl")

        print(f"\nCV ROC-AUC: {resultado_cv['media']:.4f} ± {resultado_cv['desvio_padrao']:.4f}")

    executar()


if __name__ == "__main__":
    main()
