"""
Testes unitários para src/features/build_features.py
Execute com: pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features.build_features import (
    FEATURES_CATEGORICAS,
    FEATURES_NUMERICAS,
    ChurnFeatureEngineer,
    construir_pipeline_completo,
    construir_preprocessador,
)


@pytest.fixture
def X_exemplo() -> pd.DataFrame:
    """Matriz de features sintética (pós-limpeza, pré-engenharia de features)."""
    return pd.DataFrame(
        {
            "gender": ["Male", "Female"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "No"],
            "tenure": [12, 48],
            "PhoneService": ["Yes", "Yes"],
            "MultipleLines": ["No", "Yes"],
            "InternetService": ["DSL", "Fiber optic"],
            "OnlineSecurity": ["No", "Yes"],
            "OnlineBackup": ["Yes", "No"],
            "DeviceProtection": ["No", "Yes"],
            "TechSupport": ["No", "Yes"],
            "StreamingTV": ["No", "Yes"],
            "StreamingMovies": ["No", "No"],
            "Contract": ["Month-to-month", "Two year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Bank transfer (automatic)"],
            "MonthlyCharges": [29.85, 79.95],
            "TotalCharges": [358.2, 3839.6],
        }
    )


class TestChurnFeatureEngineer:
    def test_adiciona_n_servicos(self, X_exemplo):
        transformado = ChurnFeatureEngineer().fit_transform(X_exemplo)
        assert "n_servicos" in transformado.columns

    def test_valores_corretos_n_servicos(self, X_exemplo):
        transformado = ChurnFeatureEngineer().fit_transform(X_exemplo)
        # Linha 0: OnlineBackup=Yes -> 1 serviço
        # Linha 1: Security, DevProtection, TechSupport, StreamingTV = 4
        assert transformado.loc[0, "n_servicos"] == 1
        assert transformado.loc[1, "n_servicos"] == 4

    def test_adiciona_gasto_mensal_medio(self, X_exemplo):
        transformado = ChurnFeatureEngineer().fit_transform(X_exemplo)
        assert "gasto_mensal_medio" in transformado.columns

    def test_formula_gasto_mensal_medio(self, X_exemplo):
        transformado = ChurnFeatureEngineer().fit_transform(X_exemplo)
        esperado = X_exemplo["TotalCharges"] / (X_exemplo["tenure"] + 1)
        pd.testing.assert_series_equal(
            transformado["gasto_mensal_medio"].reset_index(drop=True),
            esperado.reset_index(drop=True),
            check_names=False,
        )

    def test_adiciona_contrato_longo_prazo(self, X_exemplo):
        transformado = ChurnFeatureEngineer().fit_transform(X_exemplo)
        assert "contrato_longo_prazo" in transformado.columns
        assert transformado.loc[0, "contrato_longo_prazo"] == 0   # Month-to-month
        assert transformado.loc[1, "contrato_longo_prazo"] == 1   # Two year

    def test_adiciona_grupo_tenure(self, X_exemplo):
        transformado = ChurnFeatureEngineer().fit_transform(X_exemplo)
        assert "grupo_tenure" in transformado.columns
        assert transformado.loc[0, "grupo_tenure"] == "0-12m"   # tenure=12
        assert transformado.loc[1, "grupo_tenure"] == "25-48m"  # tenure=48

    def test_nao_modifica_original(self, X_exemplo):
        colunas_originais = set(X_exemplo.columns)
        ChurnFeatureEngineer().fit_transform(X_exemplo)
        assert set(X_exemplo.columns) == colunas_originais


class TestConstruirPreprocessador:
    def test_retorna_column_transformer(self, X_exemplo):
        from sklearn.compose import ColumnTransformer
        preprocessador = construir_preprocessador()
        assert isinstance(preprocessador, ColumnTransformer)

    def test_ajusta_e_transforma_sem_erro(self, X_exemplo):
        X_eng = ChurnFeatureEngineer().fit_transform(X_exemplo)
        preprocessador = construir_preprocessador()
        resultado = preprocessador.fit_transform(X_eng)
        assert resultado.shape[0] == len(X_exemplo)
        assert resultado.shape[1] > 0

    def test_saida_sem_nans(self, X_exemplo):
        X_eng = ChurnFeatureEngineer().fit_transform(X_exemplo)
        preprocessador = construir_preprocessador()
        resultado = preprocessador.fit_transform(X_eng)
        assert not np.isnan(resultado).any()


class TestPipelineCompleto:
    def test_pipeline_tem_tres_etapas(self):
        from sklearn.linear_model import LogisticRegression
        pipe = construir_pipeline_completo(LogisticRegression())
        assert len(pipe.steps) == 3

    def test_pipeline_treina_e_prediz(self, X_exemplo):
        from sklearn.linear_model import LogisticRegression
        y = pd.Series([0, 1])
        pipe = construir_pipeline_completo(LogisticRegression(max_iter=1000))
        pipe.fit(X_exemplo, y)
        predicoes = pipe.predict(X_exemplo)
        assert len(predicoes) == len(X_exemplo)
        assert set(predicoes).issubset({0, 1})
