import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from typing import Dict, Optional, Tuple
import joblib
import os


class TLearner:
    """T-Learner: Two separate models for treated and control groups."""

    def __init__(self, config: dict):
        self.config = config
        self.t_config = config["uplift"]["t_learner"]
        self.model_treated = None
        self.model_control = None
        self.is_fitted = False

    def _build_model(self):
        """Initialize base model."""
        if self.t_config["base_model"] == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=self.t_config["n_estimators"],
                max_depth=self.t_config["max_depth"],
                learning_rate=self.t_config["learning_rate"],
                random_state=42,
            )
        elif self.t_config["base_model"] == "random_forest":
            return RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        else:
            return LinearRegression()

    def fit(self, X: pd.DataFrame, treatment: pd.Series, outcome: pd.Series) -> Dict:
        """Fit T-Learner."""
        treated_mask = treatment == 1
        control_mask = treatment == 0

        self.model_treated = self._build_model()
        self.model_control = self._build_model()

        self.model_treated.fit(X[treated_mask], outcome[treated_mask])
        self.model_control.fit(X[control_mask], outcome[control_mask])
        self.is_fitted = True

        # Training metrics
        pred_treated = self.model_treated.predict(X[treated_mask])
        pred_control = self.model_control.predict(X[control_mask])
        return {
            "train_mse_treated": round(float(np.mean((outcome[treated_mask] - pred_treated) ** 2)), 6),
            "train_mse_control": round(float(np.mean((outcome[control_mask] - pred_control) ** 2)), 6),
        }

    def predict_individual_treatment_effect(self, X: pd.DataFrame) -> np.ndarray:
        """ITE = E[Y|X,T=1] - E[Y|X,T=0]"""
        return self.model_treated.predict(X) - self.model_control.predict(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict ITE."""
        return self.predict_individual_treatment_effect(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get combined feature importance."""
        imp_t = self.model_treated.feature_importances_
        imp_c = self.model_control.feature_importances_
        importance = (imp_t + imp_c) / 2
        return pd.DataFrame({
            "feature": range(len(importance)),
            "importance": importance,
        }).sort_values("importance", ascending=False)

    def save_model(self, path: str):
        """Save models."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump({"treated": self.model_treated, "control": self.model_control}, path)

    def load_model(self, path: str):
        """Load models."""
        data = joblib.load(path)
        self.model_treated = data["treated"]
        self.model_control = data["control"]
        self.is_fitted = True


class SLearner:
    """S-Learner: Single model with treatment as a feature."""

    def __init__(self, config: dict):
        self.config = config
        self.s_config = config["uplift"]["s_learner"]
        self.model = None
        self.is_fitted = False

    def _build_model(self):
        """Initialize base model."""
        if self.s_config["base_model"] == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=self.s_config["n_estimators"],
                max_depth=self.s_config["max_depth"],
                learning_rate=self.s_config["learning_rate"],
                random_state=42,
            )
        else:
            return LinearRegression()

    def fit(self, X: pd.DataFrame, treatment: pd.Series, outcome: pd.Series) -> Dict:
        """Fit S-Learner."""
        X_with_treatment = X.copy()
        X_with_treatment["treatment"] = treatment.values

        self.model = self._build_model()
        self.model.fit(X_with_treatment, outcome)
        self.is_fitted = True

        pred = self.model.predict(X_with_treatment)
        return {
            "train_mse": round(float(np.mean((outcome - pred) ** 2)), 6),
        }

    def predict_individual_treatment_effect(self, X: pd.DataFrame) -> np.ndarray:
        """ITE = f(X, T=1) - f(X, T=0)"""
        X_treated = X.copy()
        X_treated["treatment"] = 1
        X_control = X.copy()
        X_control["treatment"] = 0
        return self.model.predict(X_treated) - self.model.predict(X_control)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict ITE."""
        return self.predict_individual_treatment_effect(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        importance = self.model.feature_importances_
        feature_names = list(self.model.feature_names_in_) if hasattr(self.model, "feature_names_in_") else range(len(importance))
        return pd.DataFrame({
            "feature": feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False)

    def save_model(self, path: str):
        """Save model."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump(self.model, path)

    def load_model(self, path: str):
        """Load model."""
        self.model = joblib.load(path)
        self.is_fitted = True


class XLearner:
    """X-Learner: Cross-estimation learner with propensity weighting."""

    def __init__(self, config: dict):
        self.config = config
        self.x_config = config["uplift"]["x_learner"]
        self.model_treated = None
        self.model_control = None
        self.model_imputation = None
        self.propensity_model = None
        self.is_fitted = False

    def _build_model(self):
        """Initialize base model."""
        return GradientBoostingRegressor(
            n_estimators=self.x_config["n_estimators"],
            max_depth=self.x_config["max_depth"],
            learning_rate=self.x_config["learning_rate"],
            random_state=42,
        )

    def fit(self, X: pd.DataFrame, treatment: pd.Series, outcome: pd.Series) -> Dict:
        """Fit X-Learner."""
        treated_mask = treatment == 1
        control_mask = treatment == 0

        # Stage 1: Estimate conditional expectations
        self.model_treated = self._build_model()
        self.model_control = self._build_model()
        self.model_treated.fit(X[treated_mask], outcome[treated_mask])
        self.model_control.fit(X[control_mask], outcome[control_mask])

        # Stage 2: Impute individual treatment effects
        imputed_t = outcome[treated_mask] - self.model_control.predict(X[treated_mask])
        imputed_c = self.model_treated.predict(X[control_mask]) - outcome[control_mask]

        # Stage 3: Learn imputed effects with propensity weighting
        # Estimate propensity scores
        self.propensity_model = LogisticRegression(max_iter=1000, random_state=42)
        self.propensity_model.fit(X, treatment)
        propensity_scores = np.clip(self.propensity_model.predict_proba(X)[:, 1], 0.01, 0.99)

        # Imputation model
        X_imputed = X.copy()
        self.model_imputation = self._build_model()

        # Fit on both treated and control imputed effects
        X_t_imputed = pd.concat([X[treated_mask], X[control_mask]])
        y_imputed = np.concatenate([imputed_t.values, imputed_c.values])
        weights = np.concatenate([
            propensity_scores[treated_mask],
            1 - propensity_scores[control_mask],
        ])
        self.model_imputation.fit(X_imputed, y_imputed, sample_weight=weights)
        self.is_fitted = True

        # Training metrics
        pred_treated = self.model_treated.predict(X[treated_mask])
        pred_control = self.model_control.predict(X[control_mask])
        return {
            "train_mse_treated": round(float(np.mean((outcome[treated_mask] - pred_treated) ** 2)), 6),
            "train_mse_control": round(float(np.mean((outcome[control_mask] - pred_control) ** 2)), 6),
            "propensity_mean": round(float(propensity_scores.mean()), 4),
        }

    def predict_individual_treatment_effect(self, X: pd.DataFrame) -> np.ndarray:
        """ITE using imputation model with propensity weighting."""
        return self.model_imputation.predict(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict ITE."""
        return self.predict_individual_treatment_effect(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from imputation model."""
        importance = self.model_imputation.feature_importances_
        feature_names = list(self.model_imputation.feature_names_in_) if hasattr(self.model_imputation, "feature_names_in_") else range(len(importance))
        return pd.DataFrame({
            "feature": feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False)

    def save_model(self, path: str):
        """Save models."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump({
            "treated": self.model_treated,
            "control": self.model_control,
            "imputation": self.model_imputation,
            "propensity": self.propensity_model,
        }, path)

    def load_model(self, path: str):
        """Load models."""
        data = joblib.load(path)
        self.model_treated = data["treated"]
        self.model_control = data["control"]
        self.model_imputation = data["imputation"]
        self.propensity_model = data["propensity"]
        self.is_fitted = True
