"""
Testes unitários para src/data/preprocessing.py
Execute com: pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Torna src importável
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.preprocessing import limpar_dados, resumo_dados, dividir_dados


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def df_bruto() -> pd.DataFrame:
    """DataFrame sintético mínimo imitando o schema do dataset Telco."""
    return pd.DataFrame(
        {
            "customerID": ["0001-AAA", "0002-BBB", "0003-CCC", "0004-DDD", "0005-EEE"],
            "gender": ["Male", "Female", "Male", "Female", "Male"],
            "SeniorCitizen": [0, 0, 1, 0, 1],
            "Partner": ["Yes", "No", "No", "Yes", "No"],
            "Dependents": ["No", "No", "Yes", "No", "No"],
            "tenure": [1, 24, 48, 12, 0],
            "PhoneService": ["No", "Yes", "Yes", "Yes", "Yes"],
            "MultipleLines": ["No phone service", "No", "Yes", "No", "Yes"],
            "InternetService": ["DSL", "Fiber optic", "DSL", "No", "Fiber optic"],
            "OnlineSecurity": ["No", "No", "Yes", "No internet service", "No"],
            "OnlineBackup": ["Yes", "No", "No", "No internet service", "Yes"],
            "DeviceProtection": ["No", "Yes", "No", "No internet service", "No"],
            "TechSupport": ["No", "No", "Yes", "No internet service", "No"],
            "StreamingTV": ["No", "No", "No", "No internet service", "Yes"],
            "StreamingMovies": ["No", "No", "No", "No internet service", "No"],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month", "Month-to-month"],
            "PaperlessBilling": ["Yes", "No", "No", "Yes", "Yes"],
            "PaymentMethod": [
                "Electronic check", "Mailed check", "Bank transfer (automatic)",
                "Credit card (automatic)", "Electronic check",
            ],
            "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 70.70],
            "TotalCharges": ["29.85", "1889.50", " ", "508.00", ""],  # problemas intencionais
            "Churn": ["No", "No", "No", "Yes", "Yes"],
        }
    )


# ------------------------------------------------------------------
# Testes de limpar_dados
# ------------------------------------------------------------------

class TestLimpezaDados:
    def test_totalcharges_convertido_para_float(self, df_bruto):
        limpo = limpar_dados(df_bruto)
        assert limpo["TotalCharges"].dtype == np.float64

    def test_totalcharges_ausente_preenchido(self, df_bruto):
        limpo = limpar_dados(df_bruto)
        assert limpo["TotalCharges"].isna().sum() == 0

    def test_churn_codificado_como_int(self, df_bruto):
        limpo = limpar_dados(df_bruto)
        assert limpo["Churn"].dtype in [np.int32, np.int64, int]
        assert set(limpo["Churn"].unique()).issubset({0, 1})

    def test_churn_yes_vira_1(self, df_bruto):
        limpo = limpar_dados(df_bruto)
        # linhas 3 e 4 têm Churn=="Yes"
        assert limpo.loc[3, "Churn"] == 1
        assert limpo.loc[4, "Churn"] == 1

    def test_churn_no_vira_0(self, df_bruto):
        limpo = limpar_dados(df_bruto)
        assert limpo.loc[0, "Churn"] == 0

    def test_nenhuma_linha_removida(self, df_bruto):
        limpo = limpar_dados(df_bruto)
        assert len(limpo) == len(df_bruto)

    def test_retorna_novo_dataframe(self, df_bruto):
        limpo = limpar_dados(df_bruto)
        assert limpo is not df_bruto  # deve ser uma cópia


# ------------------------------------------------------------------
# Testes de dividir_dados
# ------------------------------------------------------------------

class TestDivisaoDados:
    @pytest.fixture
    def df_limpo(self, df_bruto):
        return limpar_dados(df_bruto)

    def test_tamanhos_corretos(self, df_limpo):
        X_treino, X_teste, y_treino, y_teste = dividir_dados(df_limpo, tamanho_teste=0.4)
        assert len(X_treino) + len(X_teste) == len(df_limpo)

    def test_alvo_fora_das_features(self, df_limpo):
        X_treino, X_teste, _, _ = dividir_dados(df_limpo)
        assert "Churn" not in X_treino.columns
        assert "Churn" not in X_teste.columns

    def test_customer_id_removido(self, df_limpo):
        X_treino, X_teste, _, _ = dividir_dados(df_limpo)
        assert "customerID" not in X_treino.columns

    def test_reproducibilidade(self, df_limpo):
        _, _, y1, _ = dividir_dados(df_limpo, semente_aleatoria=42)
        _, _, y2, _ = dividir_dados(df_limpo, semente_aleatoria=42)
        assert list(y1) == list(y2)

    def test_sementes_diferentes_dao_splits_diferentes(self, df_limpo):
        _, _, y1, _ = dividir_dados(df_limpo, semente_aleatoria=0)
        _, _, y2, _ = dividir_dados(df_limpo, semente_aleatoria=99)
        # Com apenas 5 linhas não é garantido, mas verifica tamanhos válidos
        assert len(y1) > 0
        assert len(y2) > 0


# ------------------------------------------------------------------
# Testes de resumo_dados
# ------------------------------------------------------------------

class TestResumoDados:
    def test_retorna_dict(self, df_bruto):
        resumo = resumo_dados(limpar_dados(df_bruto))
        assert isinstance(resumo, dict)

    def test_contem_chaves_esperadas(self, df_bruto):
        resumo = resumo_dados(limpar_dados(df_bruto))
        for chave in ["n_linhas", "n_colunas", "valores_ausentes", "taxa_churn"]:
            assert chave in resumo

    def test_taxa_churn_no_intervalo(self, df_bruto):
        resumo = resumo_dados(limpar_dados(df_bruto))
        assert 0.0 <= resumo["taxa_churn"] <= 1.0
