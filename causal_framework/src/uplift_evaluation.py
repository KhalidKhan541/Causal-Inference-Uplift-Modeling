import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")


class UpliftEvaluator:
    """Evaluate uplift models using Qini, AUUC, and uplift curves."""

    def __init__(self, config: dict):
        self.config = config

    def qini_curve(self, y_true: np.ndarray, treatment: np.ndarray,
                   uplift_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Qini curve."""
        n = len(y_true)
        order = np.argsort(-uplift_scores)
        y_sorted = y_true[order]
        t_sorted = treatment[order]

        # Cumulative counts
        n_treated = np.cumsum(t_sorted)
        n_control = np.cumsum(1 - t_sorted)

        # Cumulative outcomes
        y_treated = np.cumsum(y_sorted * t_sorted)
        y_control = np.cumsum(y_sorted * (1 - t_sorted))

        # Qini = treatment - control (normalized)
        qini = np.zeros(n)
        for i in range(n):
            if n_treated[i] > 0 and n_control[i] > 0:
                rate_t = y_treated[i] / n_treated[i]
                rate_c = y_control[i] / n_control[i]
                qini[i] = (rate_t - rate_c) * (n_treated[i] + n_control[i]) / n
            else:
                qini[i] = 0 if i == 0 else qini[i - 1]

        random_qini = np.linspace(0, qini[-1], n)
        return qini, random_qini

    def qini_coefficient(self, y_true: np.ndarray, treatment: np.ndarray,
                          uplift_scores: np.ndarray) -> float:
        """Compute Qini coefficient (area under Qini curve / area under random)."""
        qini, random_qini = self.qini_curve(y_true, treatment, uplift_scores)
        # Trapezoidal integration
        qini_area = np.trapz(qini)
        random_area = np.trapz(random_qini)
        if abs(random_area) < 1e-10:
            return 0.0
        return round(float(qini_area / random_area), 4)

    def auuc(self, y_true: np.ndarray, treatment: np.ndarray,
             uplift_scores: np.ndarray) -> float:
        """Area Under the Uplift Curve (AUUC)."""
        n = len(y_true)
        order = np.argsort(-uplift_scores)
        y_sorted = y_true[order]
        t_sorted = treatment[order]

        n_treated = np.cumsum(t_sorted)
        n_control = np.cumsum(1 - t_sorted)
        y_treated = np.cumsum(y_sorted * t_sorted)
        y_control = np.cumsum(y_sorted * (1 - t_sorted))

        uplift = np.zeros(n)
        for i in range(n):
            if n_treated[i] > 0 and n_control[i] > 0:
                uplift[i] = y_treated[i] / n_treated[i] - y_control[i] / n_control[i]
            else:
                uplift[i] = 0

        return round(float(np.trapz(uplift) / n), 4)

    def uplift_curve(self, y_true: np.ndarray, treatment: np.ndarray,
                     uplift_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute uplift curve."""
        n = len(y_true)
        order = np.argsort(-uplift_scores)
        y_sorted = y_true[order]
        t_sorted = treatment[order]

        n_treated = np.cumsum(t_sorted)
        n_control = np.cumsum(1 - t_sorted)
        y_treated = np.cumsum(y_sorted * t_sorted)
        y_control = np.cumsum(y_sorted * (1 - t_sorted))

        uplift = np.zeros(n)
        for i in range(n):
            if n_treated[i] > 0 and n_control[i] > 0:
                uplift[i] = (y_treated[i] / n_treated[i] - y_control[i] / n_control[i]) * (i + 1) / n
            else:
                uplift[i] = 0
        return uplift

    def evaluate_model(self, y_true: np.ndarray, treatment: np.ndarray,
                        uplift_scores: np.ndarray) -> Dict:
        """Full evaluation of uplift model."""
        qini = self.qini_coefficient(y_true, treatment, uplift_scores)
        auuc_val = self.auuc(y_true, treatment, uplift_scores)

        # Additional metrics
        n = len(y_true)
        order = np.argsort(-uplift_scores)
        top_10_pct = int(n * 0.1)
        top_treated = y_true[order[:top_10_pct]][treatment[order[:top_10_pct]] == 1]
        top_control = y_true[order[:top_10_pct]][treatment[order[:top_10_pct]] == 0]
        lift_at_10 = float(np.mean(top_treated) - np.mean(top_control)) if len(top_treated) > 0 and len(top_control) > 0 else 0

        return {
            "qini_coefficient": qini,
            "auuc": auuc_val,
            "lift_at_10pct": round(lift_at_10, 4),
            "n_samples": n,
            "n_treated": int(treatment.sum()),
            "n_control": int((1 - treatment).sum()),
            "mean_uplift_score": round(float(uplift_scores.mean()), 4),
            "std_uplift_score": round(float(uplift_scores.std()), 4),
        }

    def cross_validate_models(self, models: Dict[str, object], X: pd.DataFrame,
                               treatment: pd.Series, outcome: pd.Series,
                               cv_folds: int = 5) -> pd.DataFrame:
        """Cross-validate multiple uplift models."""
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        results = []

        for model_name, model in models.items():
            fold_scores = []
            for train_idx, val_idx in skf.split(X, treatment):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                t_train, t_val = treatment.iloc[train_idx], treatment.iloc[val_idx]
                y_train, y_val = outcome.iloc[train_idx], outcome.iloc[val_idx]

                # Clone and fit
                import copy
                m = copy.deepcopy(model)
                m.fit(X_train, t_train, y_train)
                ite = m.predict(X_val)
                score = self.qini_coefficient(y_val.values, t_val.values, ite)
                fold_scores.append(score)

            results.append({
                "model": model_name,
                "mean_qini": round(float(np.mean(fold_scores)), 4),
                "std_qini": round(float(np.std(fold_scores)), 4),
                "min_qini": round(float(np.min(fold_scores)), 4),
                "max_qini": round(float(np.max(fold_scores)), 4),
            })

        return pd.DataFrame(results).sort_values("mean_qini", ascending=False)

    def select_best_model(self, evaluation_results: pd.DataFrame) -> str:
        """Select best model by Qini coefficient."""
        return evaluation_results.iloc[0]["model"]

    def segment_analysis(self, y_true: np.ndarray, treatment: np.ndarray,
                          uplift_scores: np.ndarray, segments: pd.Series) -> pd.DataFrame:
        """Analyze uplift by customer segments."""
        results = []
        for segment in segments.unique():
            mask = segments == segment
            if mask.sum() < 10:
                continue
            eval_result = self.evaluate_model(y_true[mask], treatment[mask], uplift_scores[mask])
            eval_result["segment"] = segment
            results.append(eval_result)
        return pd.DataFrame(results)
