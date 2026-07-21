import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import json
import os


class CausalReport:
    """Generate comprehensive causal inference reports."""

    def __init__(self, config: dict):
        self.config = config

    def generate_report(self, results: Dict) -> str:
        """Generate full text report."""
        r = results
        report = []
        report.append("=" * 70)
        report.append("  CAUSAL INFERENCE & UPLIFT MODELING — EXECUTIVE REPORT")
        report.append("=" * 70)
        report.append(f"\nGenerated: {datetime.now().isoformat()}\n")

        # A/B Test Results
        if "ab_test" in r:
            ab = r["ab_test"]
            report.append("─" * 70)
            report.append("  SECTION 1: A/B TEST DESIGN & RESULTS")
            report.append("─" * 70)
            report.append(f"  Sample size per group:     {ab.get('n_per_group', 'N/A'):,}")
            report.append(f"  Control conversion rate:   {ab.get('control_rate', 0)*100:.2f}%")
            report.append(f"  Treatment conversion rate: {ab.get('treatment_rate', 0)*100:.2f}%")
            report.append(f"  Absolute lift:             {ab.get('absolute_lift', 0)*100:.2f} pp")
            report.append(f"  Relative lift:             {ab.get('relative_lift_pct', 0):.1f}%")
            report.append(f"  p-value:                   {ab.get('p_value', 1):.6f}")
            report.append(f"  Significant:               {'Yes ✓' if ab.get('significant', False) else 'No ✗'}")
            report.append(f"  95% CI:                    [{ab.get('ci_lower', 0)*100:.2f}%, {ab.get('ci_upper', 0)*100:.2f}%]")
            report.append("")

        # Power Analysis
        if "power_analysis" in r:
            pa = r["power_analysis"]
            report.append("─" * 70)
            report.append("  SECTION 2: POWER ANALYSIS")
            report.append("─" * 70)
            report.append(f"  Actual power:              {pa.get('actual_power', 0)*100:.1f}%")
            report.append(f"  Required sample size:      {pa.get('required_n_per_group', 'N/A'):,}")
            report.append(f"  MDE at current n:          {pa.get('mde_at_n', 0)*100:.2f}%")
            report.append(f"  Recommendation:            {pa.get('recommendation', 'N/A')}")
            report.append("")

        # Propensity Score Matching
        if "propensity" in r:
            ps = r["propensity"]
            report.append("─" * 70)
            report.append("  SECTION 3: PROPENSITY SCORE MATCHING")
            report.append("─" * 70)
            report.append(f"  Matched pairs:             {ps.get('matched_pairs', 0):,}")
            report.append(f"  Matched ATE:               {ps.get('matched_ate', 0):.6f}")
            report.append(f"  95% CI:                    [{ps.get('ci_lower', 0):.6f}, {ps.get('ci_upper', 0):.6f}]")
            report.append(f"  Avg match distance:        {ps.get('avg_match_distance', 0):.6f}")
            if "balance" in ps:
                report.append("  Covariate balance (SMD):")
                for feat, bal in list(ps["balance"].items())[:5]:
                    status = "✓" if bal["balanced"] else "✗"
                    report.append(f"    {feat:25s} SMD={bal['smd']:.4f} {status}")
            report.append("")

        # DoWhy Causal
        if "dowhy" in r:
            dw = r["dowhy"]
            report.append("─" * 70)
            report.append("  SECTION 4: DoWhy CAUSAL INFERENCE")
            report.append("─" * 70)
            report.append(f"  Estimated ATE:             {dw.get('ate', 0):.6f}")
            report.append(f"  Method:                    {dw.get('method', 'N/A')}")
            report.append(f"  Confounders adjusted:      {', '.join(dw.get('confounders', []))}")
            if dw.get("refutations"):
                report.append("  Refutation tests:")
                for test, res in dw["refutations"].items():
                    report.append(f"    {test:25s} p={res.get('p_value', 'N/A')}")
            report.append("")

        # Uplift Models
        if "uplift" in r:
            up = r["uplift"]
            report.append("─" * 70)
            report.append("  SECTION 5: UPLIFT MODEL COMPARISON")
            report.append("─" * 70)
            if isinstance(up, pd.DataFrame):
                report.append(f"  {'Model':<20s} {'Qini':>8s} {'± Std':>8s}")
                report.append(f"  {'─'*20} {'─'*8} {'─'*8}")
                for _, row in up.iterrows():
                    report.append(f"  {row['model']:<20s} {row['mean_qini']:>8.4f} {row['std_qini']:>8.4f}")
            report.append(f"\n  Best model:               {up.iloc[0]['model'] if isinstance(up, pd.DataFrame) else 'N/A'}")
            report.append("")

        # Business Recommendations
        report.append("─" * 70)
        report.append("  SECTION 6: BUSINESS RECOMMENDATIONS")
        report.append("─" * 70)
        recs = self._generate_recommendations(results)
        for i, rec in enumerate(recs, 1):
            report.append(f"  {i}. {rec}")
        report.append("")
        report.append("=" * 70)
        report.append("  END OF REPORT")
        report.append("=" * 70)

        return "\n".join(report)

    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate actionable business recommendations."""
        recs = []
        if "ab_test" in r := results.get("ab_test", {}):
            if r.get("significant"):
                recs.append(f"Deploy treatment — significant lift of {r.get('relative_lift_pct', 0):.1f}% detected.")
            else:
                recs.append("Do not deploy — results not statistically significant. Consider larger sample.")
        if "power_analysis" in r := results.get("power_analysis", {}):
            if not r.get("power_sufficient"):
                recs.append(f"Increase sample size — current power is {r.get('actual_power', 0)*100:.0f}% (target: 80%).")
        if "uplift" in r := results.get("uplift", None):
            if isinstance(r, pd.DataFrame) and len(r) > 0:
                best = r.iloc[0]["model"]
                qini = r.iloc[0]["mean_qini"]
                recs.append(f"Use {best} model for targeting (Qini={qini:.4f}).")
                recs.append("Target customers with highest uplift scores — they respond best to intervention.")
        if "propensity" in r := results.get("propensity", {}):
            if r.get("matched_pairs", 0) > 0:
                recs.append("Propensity matching confirms causal effect — observational data is valid for analysis.")
        if not recs:
            recs.append("Review data quality and re-run analysis with larger sample.")
        return recs

    def save_report(self, results: Dict, output_dir: str) -> str:
        """Save report to file."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Text report
        txt_path = os.path.join(output_dir, f"causal_report_{timestamp}.txt")
        report_text = self.generate_report(results)
        with open(txt_path, "w") as f:
            f.write(report_text)
        # JSON
        json_path = os.path.join(output_dir, f"causal_report_{timestamp}.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        return txt_path
