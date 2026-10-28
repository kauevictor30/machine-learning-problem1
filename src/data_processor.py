from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class PreprocessingReport:
    total_rows: int
    total_cols: int
    top_correlations: pd.Series
    null_summary: pd.DataFrame
    numeric_imputed_col: str
    numeric_null_count: int
    numeric_fill_value: float
    categorical_imputed_col: str
    categorical_null_count: int
    categorical_fill_value: str
    features_before_encoding: int
    features_after_encoding: int


class HousePricesDataProcessor:
    """Service responsible for EDA statistics, missing value imputation, and domain feature engineering."""

    QUALITY_MAPPING: Dict[str, int] = {
        "None": 0,
        "Po": 1,
        "Fa": 2,
        "TA": 3,
        "Gd": 4,
        "Ex": 5,
    }

    ORDINAL_QUALITY_COLS: List[str] = [
        "ExterQual",
        "ExterCond",
        "BsmtQual",
        "BsmtCond",
        "HeatingQC",
        "KitchenQual",
        "FireplaceQu",
        "GarageQual",
        "GarageCond",
        "PoolQC",
    ]

    # Columns where NA in Ames Housing means physical absence of the feature
    ABSENCE_CATEGORICAL_COLS: List[str] = [
        "PoolQC",
        "MiscFeature",
        "Alley",
        "Fence",
        "MasVnrType",
        "FireplaceQu",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "BsmtExposure",
        "BsmtFinType2",
        "BsmtQual",
        "BsmtCond",
        "BsmtFinType1",
    ]

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.numeric_medians_: Dict[str, float] = {}
        self.categorical_modes_: Dict[str, str] = {}
        self.encoded_columns_: List[str] = []
        self.neigh_ppsf_map_: Dict[str, float] = {}
        self.global_ppsf_: float = 0.0

    def load_raw_data(self) -> pd.DataFrame:
        return pd.read_csv(self.csv_path)

    def analyze_correlations_and_nulls(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        numeric_df = df.select_dtypes(include=[np.number]).drop(columns=["Id"], errors="ignore")
        correlations = numeric_df.corr()["SalePrice"].drop("SalePrice").sort_values(ascending=False)

        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0].sort_values(ascending=False)
        null_summary = pd.DataFrame(
            {
                "Coluna": null_cols.index,
                "Tipo": [str(df[c].dtype) for c in null_cols.index],
                "Qtd_Nulos": null_cols.values,
                "Percentual (%)": np.round((null_cols.values / len(df)) * 100, 2),
            }
        ).reset_index(drop=True)

        return correlations, null_summary

    @staticmethod
    def _add_domain_features(encoded_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
        out = encoded_df.copy()
        total_sf = raw_df["TotalBsmtSF"].fillna(0) + raw_df["1stFlrSF"].fillna(0) + raw_df["2ndFlrSF"].fillna(0)
        total_bath = (
            raw_df["FullBath"].fillna(0)
            + 0.5 * raw_df["HalfBath"].fillna(0)
            + raw_df["BsmtFullBath"].fillna(0)
            + 0.5 * raw_df["BsmtHalfBath"].fillna(0)
        )
        total_porch = (
            raw_df["OpenPorchSF"].fillna(0)
            + raw_df["3SsnPorch"].fillna(0)
            + raw_df["EnclosedPorch"].fillna(0)
            + raw_df["ScreenPorch"].fillna(0)
            + raw_df["WoodDeckSF"].fillna(0)
        )

        out["TotalSF"] = total_sf
        out["TotalBath"] = total_bath
        out["TotalPorchSF"] = total_porch
        out["HouseAge"] = np.maximum(0, raw_df["YrSold"] - raw_df["YearBuilt"])
        out["RemodAge"] = np.maximum(0, raw_df["YrSold"] - raw_df["YearRemodAdd"])
        out["IsNew"] = (raw_df["YrSold"] == raw_df["YearBuilt"]).astype(int)
        out["OverallQual_Sq"] = raw_df["OverallQual"] ** 2
        out["OverallQual_Cub"] = raw_df["OverallQual"] ** 3
        out["Qual_x_TotalSF"] = raw_df["OverallQual"] * total_sf
        out["Qual_x_GrLivArea"] = raw_df["OverallQual"] * raw_df["GrLivArea"]
        out["Qual_x_GarageCars"] = raw_df["OverallQual"] * raw_df["GarageCars"].fillna(0)
        out["Qual_x_TotalBath"] = raw_df["OverallQual"] * total_bath
        out["ExterQual_x_TotalSF"] = out["ExterQual"] * total_sf
        out["KitchenQual_x_TotalSF"] = out["KitchenQual"] * total_sf
        out["BsmtQual_x_TotalBsmtSF"] = out["BsmtQual"] * raw_df["TotalBsmtSF"].fillna(0)
        out["Log_GrLivArea"] = np.log1p(raw_df["GrLivArea"])
        out["Log_TotalSF"] = np.log1p(total_sf)
        out["Log_LotArea"] = np.log1p(raw_df["LotArea"])
        return out

    def fit_preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, PreprocessingReport]:
        correlations, null_summary = self.analyze_correlations_and_nulls(df)

        work_df = df.copy()
        y = work_df["SalePrice"].copy()
        work_df = work_df.drop(columns=["Id", "SalePrice"], errors="ignore")

        lot_nulls = int(work_df["LotFrontage"].isnull().sum())
        lot_median = float(work_df["LotFrontage"].median())

        garage_nulls = int(work_df["GarageType"].isnull().sum())
        garage_fill = "None"

        for col in self.ABSENCE_CATEGORICAL_COLS:
            if col in work_df.columns:
                work_df[col] = work_df[col].fillna("None")

        cat_cols = work_df.select_dtypes(include=["object", "string"]).columns.tolist()
        for col in cat_cols:
            mode_val = str(work_df[col].mode(dropna=True).iloc[0])
            self.categorical_modes_[col] = mode_val
            work_df[col] = work_df[col].fillna(mode_val)

        num_cols = work_df.select_dtypes(include=[np.number]).columns.tolist()
        for col in num_cols:
            median_val = float(work_df[col].median())
            self.numeric_medians_[col] = median_val
            work_df[col] = work_df[col].fillna(median_val)

        features_before = work_df.shape[1]

        for col in self.ORDINAL_QUALITY_COLS:
            if col in work_df.columns:
                work_df[col] = work_df[col].map(self.QUALITY_MAPPING).fillna(0).astype(int)

        remaining_cat_cols = work_df.select_dtypes(include=["object", "string"]).columns.tolist()
        work_df = pd.get_dummies(work_df, columns=remaining_cat_cols, drop_first=True, dtype=int)

        work_df = self._add_domain_features(work_df, df)

        self.encoded_columns_ = work_df.columns.tolist()
        features_after = work_df.shape[1] + 3  # Including the 3 Hedonic Neighborhood features added during training

        report = PreprocessingReport(
            total_rows=len(df),
            total_cols=df.shape[1],
            top_correlations=correlations,
            null_summary=null_summary,
            numeric_imputed_col="LotFrontage",
            numeric_null_count=lot_nulls,
            numeric_fill_value=lot_median,
            categorical_imputed_col="GarageType",
            categorical_null_count=garage_nulls,
            categorical_fill_value=garage_fill,
            features_before_encoding=features_before,
            features_after_encoding=features_after,
        )

        return work_df, y, report

    def fit_hedonic_neighborhood_index(
        self, X_train: pd.DataFrame, y_train: pd.Series, raw_neighborhood_train: pd.Series
    ) -> None:
        ppsf_train = y_train / X_train["TotalSF"]
        self.global_ppsf_ = float(ppsf_train.median())
        self.neigh_ppsf_map_ = ppsf_train.groupby(raw_neighborhood_train).median().to_dict()

    def apply_hedonic_neighborhood_index(
        self, X: pd.DataFrame, raw_df: pd.DataFrame
    ) -> pd.DataFrame:
        out = X.copy()
        neigh_ppsf = raw_df["Neighborhood"].map(self.neigh_ppsf_map_).fillna(self.global_ppsf_)
        out["Neigh_PPSF"] = neigh_ppsf
        out["Expected_Size_Val"] = out["TotalSF"] * neigh_ppsf
        out["Expected_Qual_Val"] = out["Expected_Size_Val"] * (raw_df["OverallQual"] / 6.0)
        return out

    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform unseen records (such as the 5 secret competition houses) using fitted parameters."""
        work_df = df.copy()
        work_df = work_df.drop(columns=["Id", "SalePrice"], errors="ignore")

        for col in self.ABSENCE_CATEGORICAL_COLS:
            if col in work_df.columns:
                work_df[col] = work_df[col].fillna("None")

        for col, mode_val in self.categorical_modes_.items():
            if col in work_df.columns:
                work_df[col] = work_df[col].fillna(mode_val)

        for col, median_val in self.numeric_medians_.items():
            if col in work_df.columns:
                work_df[col] = work_df[col].fillna(median_val)

        for col in self.ORDINAL_QUALITY_COLS:
            if col in work_df.columns:
                work_df[col] = work_df[col].map(self.QUALITY_MAPPING).fillna(0).astype(int)

        remaining_cat_cols = work_df.select_dtypes(include=["object", "string"]).columns.tolist()
        work_df = pd.get_dummies(work_df, columns=remaining_cat_cols, drop_first=True, dtype=int)
        work_df = self._add_domain_features(work_df, df)
        work_df = work_df.reindex(columns=self.encoded_columns_, fill_value=0)
        work_df = self.apply_hedonic_neighborhood_index(work_df, df)
        return work_df
