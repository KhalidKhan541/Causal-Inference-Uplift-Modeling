import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


class CausalPreprocessor:
    """Data preprocessing for causal inference and uplift modeling."""

    def __init__(self, config: dict):
        self.config = config
        self.treatment_col = config["data"]["treatment_column"]
        self.outcome_col = config["data"]["outcome_column"]
        self.id_col = config["data"]["customer_id_column"]
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self.feature_columns: List[str] = []

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV."""
        if filepath.endswith(".csv"):
            df = pd.read_csv(filepath)
        elif filepath.endswith(".parquet"):
            df = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values and basic cleaning."""
        df = df.drop_duplicates()
        # Impute numeric
        for col in self.config["features"]["numeric_columns"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(df[col].median())
        # Impute categorical
        for col in self.config["features"]["categorical_columns"]:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")
        # Ensure binary treatment
        if df[self.treatment_col].dtype == "object":
            df[self.treatment_col] = df[self.treatment_col].map({"treatment": 1, "control": 0, "T": 1, "C": 0, 1: 1, 0: 0})
        # Ensure binary outcome
        if df[self.outcome_col].dtype == "object":
            df[self.outcome_col] = df[self.outcome_col].map({"yes": 1, "no": 0, "True": 1, "False": 0, 1: 1, 0: 0})
        return df

    def encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Label encode categorical columns."""
        for col in self.config["features"]["categorical_columns"]:
            if col in df.columns:
                if fit:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.encoders[col] = le
                else:
                    le = self.encoders.get(col)
                    if le is not None:
                        df[col] = df[col].astype(str).map(lambda x: x if x in le.classes_ else le.classes_[0])
                        df[col] = le.transform(df[col])
        return df

    def build_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Run full preprocessing pipeline."""
        df = self.clean_data(df)
        df = self.encode_categoricals(df, fit=fit)
        # Create interaction features
        if "income" in df.columns and "total_spend" in df.columns:
            df["income_spend_ratio"] = df["total_spend"] / (df["income"] + 1)
        if "tenure_months" in df.columns and "purchase_frequency" in df.columns:
            df["tenure_frequency"] = df["tenure_months"] * df["purchase_frequency"]
        # Drop ID column
        drop_cols = [self.id_col, "date"] if self.id_col in df.columns else []
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
        return df

    def get_feature_matrix(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Extract X, treatment, and outcome."""
        feature_cols = [c for c in df.columns if c not in [self.treatment_col, self.outcome_col, self.id_col]]
        self.feature_columns = feature_cols
        X = df[feature_cols]
        treatment = df[self.treatment_col]
        outcome = df[self.outcome_col]
        return X, treatment, outcome

    def split_data(self, X: pd.DataFrame, treatment: pd.Series, outcome: pd.Series) -> Tuple:
        """Split into train/test."""
        idx_train, idx_test = train_test_split(
            X.index, test_size=self.config["data"]["test_ratio"],
            random_state=self.config["data"]["random_state"]
        )
        return (X.loc[idx_train], X.loc[idx_test],
                treatment.loc[idx_train], treatment.loc[idx_test],
                outcome.loc[idx_train], outcome.loc[idx_test])

    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Standardize numeric features."""
        numeric_cols = [c for c in X_train.select_dtypes(include=[np.number]).columns]
        scaler = StandardScaler()
        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
        self.scalers["features"] = scaler
        return X_train, X_test
