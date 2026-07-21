import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


class PropensityScoreMatcher:
    """Propensity Score Matching for causal inference from observational data."""

    def __init__(self, config: dict):
        self.config = config
        self.psm_config = config["propensity"]
        self.propensity_model = None
        self.propensity_scores: Optional[np.ndarray] = None
        self.scaler = StandardScaler()

    def _build_propensity_model(self):
        """Initialize propensity score model."""
        if self.psm_config["method"] == "logistic_regression":
            return LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        elif self.psm_config["method"] == "gradient_boosting":
            return GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
        else:
            return LogisticRegression(max_iter=1000, C=1.0, random_state=42)

    def estimate_propensity(self, X: pd.DataFrame, treatment: pd.Series) -> np.ndarray:
        """Estimate propensity scores."""
        self.propensity_model = self._build_propensity_model()
        X_scaled = self.scaler.fit_transform(X)
        self.propensity_model.fit(X_scaled, treatment)
        self.propensity_scores = self.propensity_model.predict_proba(X_scaled)[:, 1]
        return self.propensity_scores

    def check_common_support(self) -> Dict:
        """Check common support condition."""
        treated = self.propensity_scores[treatment == 1]
        control = self.propensity_scores[treatment == 0]
        min_treated = treated.min()
        max_treated = treated.max()
        min_control = control.min()
        max_control = control.max()
        overlap_min = max(min_treated, min_control)
        overlap_max = min(max_treated, max_control)
        in_overlap = (self.propensity_scores >= overlap_min) & (self.propensity_scores <= overlap_max)
        treated_in = (treatment == 1) & in_overlap
        control_in = (treatment == 0) & in_overlap
        return {
            "overlap_min": round(float(overlap_min), 4),
            "overlap_max": round(float(overlap_max), 4),
            "n_treated_in_overlap": int(treated_in.sum()),
            "n_control_in_overlap": int(control_in.sum()),
            "common_support_fraction": round(float(in_overlap.mean()), 4),
            "has_common_support": in_overlap.sum() > 100,
        }

    def match(self, X: pd.DataFrame, treatment: pd.Series, outcome: pd.Series,
              n_neighbors: int = None, caliper: float = None) -> Dict:
        """Perform propensity score matching."""
        n_neighbors = n_neighbors or self.psm_config["n_neighbors"]
        caliper = caliper or self.psm_config["caliper"]

        if self.propensity_scores is None:
            self.estimate_propensity(X, treatment)

        treated_idx = np.where(treatment == 1)[0]
        control_idx = np.where(treatment == 0)[0]

        # Fit nearest neighbors on control group scores
        control_scores = self.propensity_scores[control_idx].reshape(-1, 1)
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        nn.fit(control_scores)
        treated_scores = self.propensity_scores[treated_idx].reshape(-1, 1)
        distances, indices = nn.kneighbors(treated_scores)

        # Apply caliper filter
        matched_pairs = []
        for i, (t_idx, dist) in enumerate(zip(treated_idx, distances[:, 0])):
            if dist <= caliper:
                c_idx = control_idx[indices[i, 0]]
                matched_pairs.append((t_idx, c_idx, dist))

        if not matched_pairs:
            return {"error": "No matched pairs found within caliper"}

        t_indices = [p[0] for p in matched_pairs]
        c_indices = [p[1] for p in matched_pairs]
        avg_distance = np.mean([p[2] for p in matched_pairs])

        # Calculate matched ATE
        matched_t_outcome = outcome.values[t_indices]
        matched_c_outcome = outcome.values[c_indices]
        matched_ate = float(np.mean(matched_t_outcome - matched_c_outcome))
        se = float(np.sqrt(np.var(matched_t_outcome - matched_c_outcome) / len(matched_pairs)))
        ci_lower = matched_ate - 1.96 * se
        ci_upper = matched_ate + 1.96 * se

        # Check balance
        balance = self._check_balance(X, treatment, t_indices, c_indices)

        return {
            "matched_pairs": len(matched_pairs),
            "matched_ate": round(matched_ate, 6),
            "se": round(se, 6),
            "ci_lower": round(ci_lower, 6),
            "ci_upper": round(ci_upper, 6),
            "avg_match_distance": round(float(avg_distance), 6),
            "balance": balance,
            "treated_indices": t_indices,
            "control_indices": c_indices,
        }

    def _check_balance(self, X: pd.DataFrame, treatment: pd.Series,
                       t_idx: List[int], c_idx: List[int]) -> Dict:
        """Check covariate balance after matching."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()[:5]
        balance = {}
        for col in numeric_cols:
            treated_vals = X.iloc[t_idx][col].values
            control_vals = X.iloc[c_idx][col].values
            # Standardized mean difference
            pooled_std = np.sqrt((np.var(treated_vals) + np.var(control_vals)) / 2) + 1e-8
            smd = abs(np.mean(treated_vals) - np.mean(control_vals)) / pooled_std
            balance[col] = {
                "smd": round(float(smd), 4),
                "balanced": smd < 0.1,
                "mean_treated": round(float(np.mean(treated_vals)), 4),
                "mean_control": round(float(np.mean(control_vals)), 4),
            }
        return balance

    def get_propensity_distribution(self, treatment: pd.Series) -> Dict:
        """Get propensity score distributions for treated and control."""
        treated = self.propensity_scores[treatment == 1]
        control = self.propensity_scores[treatment == 0]
        return {
            "treated": {"mean": round(float(treated.mean()), 4), "std": round(float(treated.std()), 4),
                        "min": round(float(treated.min()), 4), "max": round(float(treated.max()), 4)},
            "control": {"mean": round(float(control.mean()), 4), "std": round(float(control.std()), 4),
                        "min": round(float(control.min()), 4), "max": round(float(control.max()), 4)},
        }

    def inverse_probability_weighting(self, X: pd.DataFrame, treatment: pd.Series,
                                       outcome: pd.Series) -> Dict:
        """Estimate ATE using Inverse Probability Weighting."""
        if self.propensity_scores is None:
            self.estimate_propensity(X, treatment)
        ps = np.clip(self.propensity_scores, 0.01, 0.99)
        weights = treatment / ps + (1 - treatment) / (1 - ps)
        ate_ipw = float(np.sum(weights * treatment * outcome) / np.sum(weights * treatment) -
                         np.sum(weights * (1 - treatment) * outcome) / np.sum(weights * (1 - treatment)))
        se = float(np.sqrt(1 / len(outcome) * np.sum(weights ** 2 * (outcome - ate_ipw) ** 2)))
        return {
            "ate_ipw": round(ate_ipw, 6),
            "se": round(se, 6),
            "ci_lower": round(ate_ipw - 1.96 * se, 6),
            "ci_upper": round(ate_ipw + 1.96 * se, 6),
            "mean_weight": round(float(np.mean(weights)), 4),
            "max_weight": round(float(np.max(weights)), 4),
        }
