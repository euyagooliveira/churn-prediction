"""
Módulo de visualização.

Funções de plotagem reutilizáveis para EDA, avaliação de modelos
e explicabilidade. Todas as funções retornam objetos Figure do matplotlib.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

# ── Estilo global ─────────────────────────────────────────────────
PALETA = {"sem_churn": "#2196F3", "churn": "#F44336", "neutro": "#90A4AE"}
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 120, "figure.facecolor": "white"})


# ------------------------------------------------------------------
# Gráficos de EDA
# ------------------------------------------------------------------

def grafico_distribuicao_churn(y: pd.Series, ax: Optional[plt.Axes] = None) -> plt.Figure:
    """Gráfico de barras com contagem e percentual de churn vs sem churn."""
    fig, ax = plt.subplots(figsize=(6, 4)) if ax is None else (ax.figure, ax)
    contagens = y.value_counts()
    cores = [PALETA["sem_churn"], PALETA["churn"]]
    barras = ax.bar(["Sem Churn", "Churn"], contagens.values, color=cores, width=0.5)
    for barra, contagem in zip(barras, contagens.values):
        pct = contagem / len(y) * 100
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height() + 30,
            f"{contagem:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
    ax.set_title("Distribuição de Churn", fontsize=14, fontweight="bold")
    ax.set_ylabel("Contagem")
    ax.set_ylim(0, contagens.max() * 1.2)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    return fig


def grafico_distribuicao_numericas(
    df: pd.DataFrame,
    colunas_numericas: List[str],
    alvo: str = "Churn",
) -> plt.Figure:
    """KDE das features numéricas separadas por status de churn."""
    n = len(colunas_numericas)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, colunas_numericas):
        for rotulo, cor in [(0, PALETA["sem_churn"]), (1, PALETA["churn"])]:
            subset = df[df[alvo] == rotulo][col].dropna()
            subset.plot.kde(
                ax=ax, color=cor,
                label="Sem Churn" if rotulo == 0 else "Churn",
                linewidth=2,
            )
        ax.set_title(col, fontweight="bold")
        ax.set_xlabel("")
        ax.legend()

    fig.suptitle(
        "Distribuição das Features Numéricas por Churn",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    return fig


def grafico_taxa_churn_categoricas(
    df: pd.DataFrame,
    colunas_cat: List[str],
    alvo: str = "Churn",
    max_colunas: int = 3,
) -> plt.Figure:
    """Gráficos de barra horizontal com taxa de churn por nível de categoria."""
    n = len(colunas_cat)
    ncols = min(n, max_colunas)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for idx, col in enumerate(colunas_cat):
        ax = axes[idx]
        taxa_churn = (
            df.groupby(col)[alvo]
            .mean()
            .sort_values(ascending=True)
            .mul(100)
        )
        barras = ax.barh(taxa_churn.index, taxa_churn.values, color=PALETA["churn"], alpha=0.8)
        for barra, val in zip(barras, taxa_churn.values):
            ax.text(val + 0.5, barra.get_y() + barra.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=9)
        ax.set_title(col, fontweight="bold")
        ax.set_xlabel("Taxa de Churn (%)")
        ax.set_xlim(0, taxa_churn.max() * 1.3)

    # Oculta eixos não utilizados
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Taxa de Churn por Categoria", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def grafico_heatmap_correlacao(df: pd.DataFrame, alvo: str = "Churn") -> plt.Figure:
    """Heatmap de correlação das features numéricas incluindo o alvo."""
    num_df = df.select_dtypes("number")
    corr = num_df.corr()
    mascara = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, mask=mascara, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Matriz de Correlação", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# Gráficos de avaliação do modelo
# ------------------------------------------------------------------

def grafico_curva_roc(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Curva ROC com anotação do AUC."""
    fig, ax = plt.subplots(figsize=(6, 5)) if ax is None else (ax.figure, ax)
    ax.plot(fpr, tpr, color=PALETA["churn"], lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Aleatório")
    ax.fill_between(fpr, tpr, alpha=0.08, color=PALETA["churn"])
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos")
    ax.set_title("Curva ROC", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def grafico_curva_precisao_recall(
    precisoes: np.ndarray,
    recalls: np.ndarray,
    avg_precision: float,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Curva Precisão-Recall com anotação do AP."""
    fig, ax = plt.subplots(figsize=(6, 5)) if ax is None else (ax.figure, ax)
    ax.plot(recalls, precisoes, color=PALETA["churn"], lw=2, label=f"AP = {avg_precision:.4f}")
    ax.fill_between(recalls, precisoes, alpha=0.08, color=PALETA["churn"])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precisão")
    ax.set_title("Curva Precisão-Recall", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def grafico_matriz_confusao(
    cm: np.ndarray,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Matriz de confusão estilizada."""
    fig, ax = plt.subplots(figsize=(5, 4)) if ax is None else (ax.figure, ax)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Sem Churn", "Churn"]
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matriz de Confusão", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def grafico_analise_threshold(
    y_verdadeiro: np.ndarray,
    y_prob: np.ndarray,
) -> plt.Figure:
    """Gráfico de precisão, recall e F1 em função do threshold de decisão."""
    from sklearn.metrics import precision_recall_curve

    p, r, t = precision_recall_curve(y_verdadeiro, y_prob)
    f1 = np.where((p + r) > 0, 2 * p * r / (p + r), 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, p[:-1], label="Precisão", color="#2196F3", lw=2)
    ax.plot(t, r[:-1], label="Recall", color="#F44336", lw=2)
    ax.plot(t, f1[:-1], label="F1", color="#4CAF50", lw=2, linestyle="--")

    melhor_t = t[np.argmax(f1[:-1])]
    ax.axvline(
        melhor_t, color="gray", linestyle=":", alpha=0.8,
        label=f"Melhor threshold F1 = {melhor_t:.2f}",
    )

    ax.set_xlabel("Threshold de Decisão")
    ax.set_ylabel("Score")
    ax.set_title("Análise de Threshold", fontsize=13, fontweight="bold")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return fig


def grafico_shap_resumo(
    shap_values, nomes_features: List[str], max_display: int = 20
) -> plt.Figure:
    """Gráfico beeswarm de resumo SHAP."""
    import shap
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, feature_names=nomes_features,
        max_display=max_display, show=False, plot_size=None,
    )
    fig = plt.gcf()
    fig.suptitle("Importância das Features (SHAP)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def salvar_figura(fig: plt.Figure, caminho: str, dpi: int = 150) -> None:
    """Salva a figura em disco."""
    fig.savefig(caminho, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# Aliases em inglês para compatibilidade
plot_churn_distribution = grafico_distribuicao_churn
plot_numerical_distributions = grafico_distribuicao_numericas
plot_categorical_churn_rate = grafico_taxa_churn_categoricas
plot_correlation_heatmap = grafico_heatmap_correlacao
plot_roc_curve = grafico_curva_roc
plot_precision_recall_curve = grafico_curva_precisao_recall
plot_confusion_matrix = grafico_matriz_confusao
plot_threshold_analysis = grafico_analise_threshold
plot_shap_summary = grafico_shap_resumo
save_figure = salvar_figura
