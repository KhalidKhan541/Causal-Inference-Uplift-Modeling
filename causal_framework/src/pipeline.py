import numpy as np
import pandas as pd
from typing import Dict, Optional
import os
import warnings
warnings.filterwarnings("ignore")

from src.preprocessing import CausalPreprocessor
from src.propensity import PropensityScoreMatcher
from src.causal_inference import DoWhyCausalAnalyzer
from src.models.uplift_models import TLearner, SLearner, XLearner
from src.uplift_evaluation import UpliftEvaluator
from src.power_analysis import PowerAnalyzer
from src.reporting import CausalReport
from src.utils.data_utils import load_config, generate_sample_data, calculate_ate


class CausalPipeline:
    """End-to-end causal inference and uplift modeling pipeline."""

    def __init__(self, config_path: str = "configs/default.yaml"):
        self.config = load_config(config_path)
        self.preprocessor = CausalPreprocessor(self.config)
        self.psm = PropensityScoreMatcher(self.config)
        self.dowhy = DoWhyCausalAnalyzer(self.config)
        self.t_learner = TLearner(self.config)
        self.s_learner = SLearner(self.config)
        self.x_learner = XLearner(self.config)
        self.evaluator = UpliftEvaluator(self.config)
        self.power_analyzer = PowerAnalyzer(self.config)
        self.reporter = CausalReport(self.config)
        self.results: Dict = {}

    def run(self, data_path: Optional[str] = None, use_sample_data: bool = False) -> Dict:
        """Execute the full causal inference pipeline."""
        print("=" * 60)
        print("  CAUSAL INFERENCE & UPLIFT MODELING PIPELINE")
        print("=" * 60)

        # Step 1: Load data
        print("\n[1/8] Loading data...")
        if use_sample_data:
            df = generate_sample_data()
            print(f"  Generated sample data: {df.shape[0]} customers")
        else:
            df = self.preprocessor.load_data(data_path)
            print(f"  Loaded data: {df.shape}")

        # Step 2: Preprocessing
        print("\n[2/8] Preprocessing...")
        df = self.preprocessor.build_features(df, fit=True)
        X, treatment, outcome = self.preprocessor.get_feature_matrix(df)
        print(f"  Features: {X.shape[1]}, Treatment rate: {treatment.mean():.1%}")

        # Step 3: A/B Test Design & Power Analysis
        print("\n[3/8] Power analysis...")
        baseline = outcome[treatment == 0].mean()
        power_result = self.power_analyzer.ab_test_sample_size_calculator(
            baseline_rate=baseline,
            mde=self.config["ab_test"]["min_detectable_effect"],
        )
        power_curve = self.power_analyzer.power_curve()
        print(f"  Baseline rate: {baseline:.1%}")
        print(f"  Required n/group: {power_result['sample_size_per_group']:,}")
        print(f"  Power at n: {self.power_analyzer.calculate_power(power_result['sample_size_per_group'], self.config['ab_test']['min_detectable_effect']):.1%}")

        # Step 4: Propensity Score Matching
        print("\n[4/8] Propensity score matching...")
        self.psm.estimate_propensity(X, treatment)
        common_support = self.psm.check_common_support()
        psm_result = self.psm.match(X, treatment, outcome)
        ipw_result = self.psm.inverse_probability_weighting(X, treatment, outcome)
        print(f"  Matched pairs: {psm_result.get('matched_pairs', 0):,}")
        print(f"  Matched ATE: {psm_result.get('matched_ate', 0):.6f}")
        print(f"  IPW ATE: {ipw_result['ate_ipw']:.6f}")

        # Step 5: DoWhy Causal Inference
        print("\n[5/8] DoWhy causal analysis...")
        confounders = self.config["features"]["confounders"]
        available_confounders = [c for c in confounders if c in df.columns]
        dowhy_result = self.dowhy.estimate_ate(
            df, self.config["data"]["treatment_column"],
            self.config["data"]["outcome_column"], available_confounders,
        )
        print(f"  ATE (DoWhy): {dowhy_result['ate']:.6f}")

        # Step 6: Uplift Models
        print("\n[6/8] Training uplift models...")
        X_train, X_test, t_train, t_test, y_train, y_test = self.preprocessor.split_data(X, treatment, outcome)
        self.t_learner.fit(X_train, t_train, y_train)
        self.s_learner.fit(X_train, t_train, y_train)
        self.x_learner.fit(X_train, t_train, y_train)

        ite_t = self.t_learner.predict(X_test)
        ite_s = self.s_learner.predict(X_test)
        ite_x = self.x_learner.predict(X_test)

        eval_t = self.evaluator.evaluate_model(y_test.values, t_test.values, ite_t)
        eval_s = self.evaluator.evaluate_model(y_test.values, t_test.values, ite_s)
        eval_x = self.evaluator.evaluate_model(y_test.values, t_test.values, ite_x)

        print(f"  T-Learner Qini: {eval_t['qini_coefficient']:.4f}")
        print(f"  S-Learner Qini: {eval_s['qini_coefficient']:.4f}")
        print(f"  X-Learner Qini: {eval_x['qini_coefficient']:.4f}")

        # Step 7: Model Comparison
        print("\n[7/8] Model comparison...")
        models = {"T-Learner": self.t_learner, "S-Learner": self.s_learner, "X-Learner": self.x_learner}
        comparison = self.evaluator.cross_validate_models(models, X, treatment, outcome)
        best_model_name = self.evaluator.select_best_model(comparison)
        print(f"  Best model: {best_model_name}")

        # Step 8: A/B Test Simulation
        print("\n[8/8] A/B test simulation...")
        control_rate = outcome[treatment == 0].mean()
        treatment_rate = outcome[treatment == 1].mean()
        ab_results = calculate_ate(outcome.values, treatment.values)
        ab_test_result = {
            "n_per_group": int(len(outcome) / 2),
            "control_rate": float(control_rate),
            "treatment_rate": float(treatment_rate),
            "absolute_lift": ab_results["mean_treated"] - ab_results["mean_control"],
            "relative_lift_pct": round((ab_results["mean_treated"] - ab_results["mean_control"]) / (control_rate + 1e-8) * 100, 2),
            "p_value": ab_results["p_value"],
            "significant": ab_results["significant"],
            "ci_lower": ab_results["ci_lower"],
            "ci_upper": ab_results["ci_upper"],
        }

        # Compile results
        self.results = {
            "ab_test": ab_test_result,
            "power_analysis": power_result,
            "propensity": psm_result,
            "dowhy": dowhy_result,
            "uplift": comparison,
            "uplift_eval": {"T-Learner": eval_t, "S-Learner": eval_s, "X-Learner": eval_x},
            "segment_analysis": self.evaluator.segment_analysis(
                y_test.values, t_test.values, ite_x,
                X_test.get("customer_segment", pd.Series(["default"] * len(X_test))),
            ) if "customer_segment" in X_test.columns else None,
        }

        # Report
        print("\nGenerating report...")
        report_path = self.reporter.save_report(self.results, self.config["output"]["reports_dir"])
        print(f"  Report saved: {report_path}")

        print("\n" + "=" * 60)
        print("  PIPELINE COMPLETE")
        print("=" * 60)
        print(f"\n  ATE (DoWhy):  {dowhy_result['ate']:.6f}")
        print(f"  Matched ATE:  {psm_result.get('matched_ate', 0):.6f}")
        print(f"  Best uplift:  {best_model_name}")
        print(f"  Qini:         {comparison.iloc[0]['mean_qini']:.4f}")

        return self.results
