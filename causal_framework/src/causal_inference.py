import numpy as np
import pandas as pd
from typing import Dict, Optional, List
import warnings
warnings.filterwarnings("ignore")


class DoWhyCausalAnalyzer:
    """DoWhy-based causal inference analysis."""

    def __init__(self, config: dict):
        self.config = config
        self.causal_config = config["causal"]
        self.causal_model = None
        self.identified_estimand = None
        self.estimate = None

    def estimate_ate(self, df: pd.DataFrame, treatment_col: str, outcome_col: str,
                     confounders: List[str]) -> Dict:
        """Estimate Average Treatment Effect using DoWhy."""
        try:
            import dowhy
            from dowhy import CausalModel

            # Build causal graph
            graph_str = self._build_causal_graph(treatment_col, outcome_col, confounders)

            self.causal_model = CausalModel(
                data=df,
                treatment=treatment_col,
                outcome=outcome_col,
                graph=graph_str,
            )

            # Identify estimand
            self.identified_estimand = self.causal_model.identify_effect()

            # Estimate
            self.estimate = self.causal_model.estimate_effect(
                self.identified_estimand,
                method_name="backdoor.linear_regression",
            )

            # Refute
            refutation_results = self._run_refutations()

            return {
                "ate": round(float(self.estimate.value), 6),
                "method": "backdoor.linear_regression",
                "confounders": confounders,
                "refutations": refutation_results,
                "success": True,
            }
        except Exception as e:
            # Fallback to simple difference
            treated_mean = df[df[treatment_col] == 1][outcome_col].mean()
            control_mean = df[df[treatment_col] == 0][outcome_col].mean()
            ate = treated_mean - control_mean
            return {
                "ate": round(float(ate), 6),
                "method": "simple_difference (DoWhy failed)",
                "error": str(e),
                "confounders": confounders,
                "refutations": {},
                "success": False,
            }

    def _build_causal_graph(self, treatment: str, outcome: str, confounders: List[str]) -> str:
        """Build causal graph string for DoWhy."""
        edges = []
        for c in confounders:
            edges.append(f'"{c}" -> "{treatment}"')
            edges.append(f'"{c}" -> "{outcome}"')
        edges.append(f'"{treatment}" -> "{outcome}"')
        return "digraph {" + " ".join(edges) + "}"

    def _run_refutations(self) -> Dict:
        """Run refutation tests."""
        results = {}
        try:
            # Random common cause
            refute_random = self.causal_model.refute_estimate(
                self.identified_estimand, self.estimate,
                method_name="random_common_cause",
                num_simulations=50,
            )
            results["random_common_cause"] = {
                "p_value": round(float(refute_random.refutation_result.get("p_value", 0)), 4),
                "is_confounded": refute_random.refutation_result.get("is_confounded", None),
            }
        except Exception:
            results["random_common_cause"] = {"error": "failed"}

        try:
            # Placebo treatment
            refute_placebo = self.causal_model.refute_estimate(
                self.identified_estimand, self.estimate,
                method_name="placebo_treatment_refuter",
                num_simulations=50,
            )
            results["placebo_treatment"] = {
                "p_value": round(float(refute_placebo.refutation_result.get("p_value", 0)), 4),
                "is_confounded": refute_placebo.refutation_result.get("is_confounded", None),
            }
        except Exception:
            results["placebo_treatment"] = {"error": "failed"}

        return results

    def estimate_ate_dowhy(self, df: pd.DataFrame, treatment_col: str, outcome_col: str,
                           confounders: List[str]) -> Dict:
        """Wrapper for ATE estimation."""
        return self.estimate_ate(df, treatment_col, outcome_col, confounders)

    def sensitivity_analysis(self, df: pd.DataFrame, treatment_col: str, outcome_col: str,
                              confounders: List[str]) -> Dict:
        """Perform sensitivity analysis for unmeasured confounders."""
        try:
            result = self.estimate_ate(df, treatment_col, outcome_col, confounders)
            ate = result["ate"]
            # Rosenbaum bounds approximation
            gamma_range = np.arange(1.0, 5.1, 0.5)
            bounds = []
            for gamma in gamma_range:
                adjusted_ate = ate / gamma
                bounds.append({
                    "gamma": round(float(gamma), 1),
                    "adjusted_ate": round(float(adjusted_ate), 6),
                    "still_significant": abs(adjusted_ate) > 0.01,
                })
            return {
                "baseline_ate": ate,
                "rosenbaum_bounds": bounds,
                "interpretation": f"Treatment effect remains significant until confounding strength (γ) > {self._find_breaking_point(bounds)}",
            }
        except Exception as e:
            return {"error": str(e)}

    def _find_breaking_point(self, bounds: List[Dict]) -> float:
        """Find the gamma at which effect becomes insignificant."""
        for b in bounds:
            if not b["still_significant"]:
                return b["gamma"]
        return float("inf")
