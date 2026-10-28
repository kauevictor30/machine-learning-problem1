from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class HousePricesVisualizer:
    """Service responsible for generating charts for the technical report."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", palette="deep")
        plt.rcParams["figure.dpi"] = 180
        plt.rcParams["savefig.dpi"] = 220
        plt.rcParams["font.size"] = 10

    def plot_correlation_matrix(self, df: pd.DataFrame) -> Path:
        numeric_df = df.select_dtypes(include=[np.number]).drop(columns=["Id"], errors="ignore")
        corr_matrix = numeric_df.corr()
        top_cols = corr_matrix["SalePrice"].sort_values(ascending=False).head(11).index

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix.loc[top_cols, top_cols],
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            cbar_kws={"label": "Coeficiente de Correlação de Pearson (r)"},
            ax=ax,
            linewidths=0.5,
        )
        ax.set_title(
            "Matriz de Correlação — Top 10 Variáveis vs Preço de Venda (SalePrice)",
            fontsize=13,
            fontweight="bold",
            pad=14,
        )
        plt.tight_layout()
        out_path = self.output_dir / "01_matriz_correlacao.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_top3_scatter(self, df: pd.DataFrame) -> Path:
        top3 = [
            ("OverallQual", "Qualidade Geral do Imóvel (OverallQual [1-10])", "#1d4ed8"),
            ("GrLivArea", "Área Útil Acima do Solo (GrLivArea [sq ft])", "#047857"),
            ("GarageCars", "Capacidade da Garagem (GarageCars [nº vagas])", "#b45309"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
        for ax, (col, label, color) in zip(axes, top3):
            r_val = df[col].corr(df["SalePrice"])
            sns.regplot(
                data=df,
                x=col,
                y="SalePrice",
                ax=ax,
                color=color,
                scatter_kws={"alpha": 0.45, "s": 28},
                line_kws={"color": "#dc2626", "linewidth": 2, "label": f"Tendência Linear (r = {r_val:.4f})"},
            )
            ax.set_title(f"{col} vs SalePrice (r = {r_val:.4f})", fontsize=11, fontweight="bold")
            ax.set_xlabel(label)
            ax.set_ylabel("Preço de Venda - SalePrice (US$)" if col == "OverallQual" else "")
            ax.legend(loc="upper left")

        fig.suptitle(
            "Questão 1: As 3 Variáveis com Maior Correlação Linear Positiva com SalePrice",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        out_path = self.output_dir / "02_scatter_top3_correlacao.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_missing_values_treatment(self, df: pd.DataFrame) -> Path:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

        # Left subplot: LotFrontage distribution before/after median imputation
        lot_raw = df["LotFrontage"].dropna()
        median_val = df["LotFrontage"].median()
        mean_val = df["LotFrontage"].mean()

        sns.histplot(lot_raw, kde=True, ax=axes[0], color="#2563eb", bins=35, alpha=0.55)
        axes[0].axvline(median_val, color="#16a34a", linestyle="--", linewidth=2.2, label=f"Mediana = {median_val:.1f} ft (Escolhida)")
        axes[0].axvline(mean_val, color="#dc2626", linestyle=":", linewidth=2.0, label=f"Média = {mean_val:.1f} ft")
        axes[0].set_title("Tratamento Numérico: LotFrontage (259 nulos)", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("Testada do Lote — LotFrontage (pés lineares)")
        axes[0].set_ylabel("Frequência de Imóveis")
        axes[0].legend()

        # Right subplot: GarageType distribution with 'None' category
        garage_series = df["GarageType"].fillna("None (Sem Garagem)")
        counts = garage_series.value_counts()
        sns.barplot(
            x=counts.values,
            y=counts.index,
            hue=counts.index,
            ax=axes[1],
            palette="viridis",
            legend=False,
        )
        axes[1].set_title("Tratamento Categórico: GarageType (79 nulos -> 'None')", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Quantidade de Imóveis")
        axes[1].set_ylabel("Tipo de Garagem (GarageType)")

        plt.tight_layout()
        out_path = self.output_dir / "03_tratamento_valores_nulos.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_model_evaluation(
        self,
        y_test: pd.Series,
        y_pred_baseline: np.ndarray,
        baseline_metrics: Dict[str, float],
        y_pred_comp: np.ndarray,
        comp_metrics: Dict[str, float],
    ) -> Path:
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), sharey=True)

        min_v = min(float(y_test.min()), float(y_pred_baseline.min()))
        max_v = max(float(y_test.max()), float(y_pred_baseline.max()))

        # Baseline Linear Regression
        axes[0].scatter(y_test, y_pred_baseline, alpha=0.55, color="#2563eb", edgecolors="w", s=40)
        axes[0].plot([min_v, max_v], [min_v, max_v], "r--", lw=2, label="Predição Perfeita (y = x)")
        axes[0].set_title(
            f"Baseline: Regressão Linear Múltipla\nR² = {baseline_metrics['r2']:.4f} | MAE = US$ {baseline_metrics['mae']:,.2f}",
            fontsize=11,
            fontweight="bold",
        )
        axes[0].set_xlabel("Preço Real — SalePrice (US$)")
        axes[0].set_ylabel("Preço Previsto pelo Modelo (US$)")
        axes[0].legend(loc="upper left")

        # Competition Model
        axes[1].scatter(y_test, y_pred_comp, alpha=0.55, color="#059669", edgecolors="w", s=40)
        axes[1].plot([min_v, max_v], [min_v, max_v], "r--", lw=2, label="Predição Perfeita (y = x)")
        axes[1].set_title(
            f"Modelo Otimizado (Competição — 5 Imóveis)\nR² = {comp_metrics['r2']:.4f} | MAE = US$ {comp_metrics['mae']:,.2f}",
            fontsize=11,
            fontweight="bold",
        )
        axes[1].set_xlabel("Preço Real — SalePrice (US$)")
        axes[1].legend(loc="upper left")

        plt.tight_layout()
        out_path = self.output_dir / "04_avaliacao_regressao_linear.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path
