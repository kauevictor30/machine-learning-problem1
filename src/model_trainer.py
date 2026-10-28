from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    VotingRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


@dataclass
class TrainingOutcome:
    baseline_metrics: Dict[str, float]
    competition_metrics: Dict[str, float]
    y_test: pd.Series
    y_pred_baseline: np.ndarray
    y_pred_competition: np.ndarray
    train_size: int
    test_size: int


class HousePricesModelTrainer:
    """Service responsible for training the Baseline Linear Regression and Competition ensemble."""

    def __init__(self, models_dir: Path, random_state: int = 42) -> None:
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

    @staticmethod
    def _compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        r2 = float(r2_score(y_true, y_pred))
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
        return {"r2": r2, "mae": mae, "rmse": rmse, "mape": mape}

    @staticmethod
    def _filter_training_anomalies(X_train: pd.DataFrame, y_train: pd.Series):
        # 1. Remove the 2 extreme partial-sale anomalies documented in Ames Housing (GrLivArea > 4000 & SalePrice < 300000)
        mask_partial = ~((X_train["GrLivArea"] > 4000) & (y_train < 300000))
        X_tr, y_tr = X_train[mask_partial], y_train[mask_partial]

        # 2. Remove extreme residual outliers in training set (> 0.35 log-price deviation)
        ridge_check = TransformedTargetRegressor(
            Pipeline([("s", RobustScaler()), ("r", Ridge(alpha=15.0))]),
            func=np.log1p,
            inverse_func=np.expm1,
        )
        ridge_check.fit(X_tr, y_tr)
        log_res = np.abs(np.log1p(y_tr) - np.log1p(ridge_check.predict(X_tr)))
        inlier_mask = log_res < 0.35
        return X_tr[inlier_mask], y_tr[inlier_mask]

    def _build_competition_ensemble(self) -> TransformedTargetRegressor:
        ridge_pipe = Pipeline([("scaler", RobustScaler()), ("ridge", Ridge(alpha=14.0))])
        lasso_pipe = Pipeline(
            [
                ("scaler", RobustScaler()),
                ("lasso", Lasso(alpha=0.00035, max_iter=5000, random_state=self.random_state)),
            ]
        )
        elastic_pipe = Pipeline(
            [
                ("scaler", RobustScaler()),
                (
                    "en",
                    ElasticNet(
                        alpha=0.00045,
                        l1_ratio=0.65,
                        max_iter=5000,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        gbr_l1 = GradientBoostingRegressor(
            n_estimators=280,
            learning_rate=0.035,
            max_depth=3,
            subsample=0.8,
            max_features="sqrt",
            loss="absolute_error",
            random_state=self.random_state,
        )
        gbr_huber = GradientBoostingRegressor(
            n_estimators=280,
            learning_rate=0.035,
            max_depth=4,
            subsample=0.8,
            max_features="sqrt",
            loss="huber",
            random_state=self.random_state,
        )
        hgb = HistGradientBoostingRegressor(
            max_iter=220,
            learning_rate=0.04,
            max_depth=5,
            l2_regularization=1.5,
            loss="absolute_error",
            random_state=self.random_state,
        )

        ensemble = VotingRegressor(
            estimators=[
                ("ridge", ridge_pipe),
                ("lasso", lasso_pipe),
                ("en", elastic_pipe),
                ("gbr_l1", gbr_l1),
                ("gbr_huber", gbr_huber),
                ("hgb", hgb),
            ],
            weights=[0.18, 0.22, 0.15, 0.15, 0.15, 0.15],
        )

        return TransformedTargetRegressor(
            regressor=ensemble,
            func=np.log1p,
            inverse_func=np.expm1,
        )

    def train_and_evaluate(
        self, X: pd.DataFrame, y: pd.Series, raw_df: pd.DataFrame, processor
    ) -> TrainingOutcome:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=self.random_state
        )

        # Fit Hedonic Neighborhood Price/SqFt Index strictly on X_train (zero data leakage)
        processor.fit_hedonic_neighborhood_index(
            X_train, y_train, raw_df.loc[X_train.index, "Neighborhood"]
        )
        X_train = processor.apply_hedonic_neighborhood_index(X_train, raw_df.loc[X_train.index])
        X_test = processor.apply_hedonic_neighborhood_index(X_test, raw_df.loc[X_test.index])

        # Filter extreme anomalies strictly on training data; keep X_test 100% untouched
        X_tr_clean, y_tr_clean = self._filter_training_anomalies(X_train, y_train)

        # 1. Baseline Multiple Linear Regression (OLS)
        baseline_model = LinearRegression()
        baseline_model.fit(X_tr_clean, y_tr_clean)
        y_pred_baseline = baseline_model.predict(X_test)
        baseline_metrics = self._compute_metrics(y_test, y_pred_baseline)

        # 2. Competition Model (6-Model Ensemble on Log-Target to minimize MAE)
        competition_model = self._build_competition_ensemble()
        competition_model.fit(X_tr_clean, y_tr_clean)
        y_pred_comp = competition_model.predict(X_test)
        comp_metrics = self._compute_metrics(y_test, y_pred_comp)

        # Refit final competition bundle on 100% of available inlier data for the 5 secret houses
        processor.fit_hedonic_neighborhood_index(X, y, raw_df["Neighborhood"])
        X_full = processor.apply_hedonic_neighborhood_index(X, raw_df)
        X_full_clean, y_full_clean = self._filter_training_anomalies(X_full, y)

        full_competition_model = self._build_competition_ensemble()
        full_competition_model.fit(X_full_clean, y_full_clean)

        full_baseline_model = LinearRegression()
        full_baseline_model.fit(X_full_clean, y_full_clean)

        bundle = {
            "processor": processor,
            "baseline_model": full_baseline_model,
            "competition_model_split": competition_model,
            "competition_model_full": full_competition_model,
            "baseline_metrics": baseline_metrics,
            "competition_metrics": comp_metrics,
        }
        joblib.dump(bundle, self.models_dir / "house_prices_bundle.joblib")

        return TrainingOutcome(
            baseline_metrics=baseline_metrics,
            competition_metrics=comp_metrics,
            y_test=y_test,
            y_pred_baseline=y_pred_baseline,
            y_pred_competition=y_pred_comp,
            train_size=len(X_train),
            test_size=len(X_test),
        )
