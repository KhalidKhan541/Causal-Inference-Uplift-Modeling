import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import norm, ttest_ind, chi2_contingency
import warnings
warnings.filterwarnings("ignore")


class PowerAnalyzer:
    """Statistical power analysis for A/B tests and causal inference."""

    def __init__(self, config: dict):
        self.config = config
        self.pa_config = config["power_analysis"]
        self.significance_level = config["ab_test"]["significance_level"]
        self.power = config["ab_test"]["power"]

    def calculate_sample_size_two_proportions(self, p1: float, p2: float,
                                               alpha: float = None, power: float = None) -> int:
        """Calculate sample size per group for two proportion comparison."""
        alpha = alpha or self.significance_level
        power = power or self.power
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)
        pooled_p = (p1 + p2) / 2
        n = ((z_alpha * np.sqrt(2 * pooled_p * (1 - pooled_p)) +
              z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (p1 - p2) ** 2
        return int(np.ceil(n))

    def calculate_sample_size_continuous(self, effect_size: float, alpha: float = None,
                                          power: float = None) -> int:
        """Calculate sample size per group for continuous outcome."""
        alpha = alpha or self.significance_level
        power = power or self.power
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        return int(np.ceil(n))

    def calculate_power(self, n_per_group: int, effect_size: float,
                         alpha: float = None) -> float:
        """Calculate statistical power for given sample size."""
        alpha = alpha or self.significance_level
        z_alpha = norm.ppf(1 - alpha / 2)
        se = np.sqrt(2 / n_per_group)
        z = (effect_size / se) - z_alpha
        return float(norm.cdf(z))

    def power_curve(self, effect_sizes: List[float] = None,
                    sample_sizes: List[int] = None) -> pd.DataFrame:
        """Compute power for different effect sizes and sample sizes."""
        effect_sizes = effect_sizes or self.pa_config["effect_sizes"]
        sample_sizes = sample_sizes or self.pa_config["sample_sizes"]
        results = []
        for n in sample_sizes:
            for es in effect_sizes:
                power = self.calculate_power(n, es)
                results.append({
                    "sample_size": n,
                    "effect_size": es,
                    "power": round(power, 4),
                    "sufficient": power >= self.power,
                })
        return pd.DataFrame(results)

    def minimum_detectable_effect(self, n_per_group: int, alpha: float = None,
                                   power: float = None) -> float:
        """Calculate minimum detectable effect size."""
        alpha = alpha or self.significance_level
        power = power or self.power
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)
        mde = (z_alpha + z_beta) * np.sqrt(2 / n_per_group)
        return round(float(mde), 4)

    def sequential_testing_boundaries(self, n_looks: int, alpha: float = None,
                                       method: str = "bonferroni") -> List[float]:
        """Calculate adjusted significance boundaries for sequential testing."""
        alpha = alpha or self.significance_level
        if method == "bonferroni":
            return [alpha / n_looks] * n_looks
        elif method == "obrien_fleming":
            boundaries = []
            for i in range(1, n_looks + 1):
                z = norm.ppf(1 - alpha / 2) * np.sqrt(n_looks / i)
                boundaries.append(2 * (1 - norm.cdf(z)))
            return boundaries
        else:
            return [alpha / n_looks] * n_looks

    def sample_ratio_mismatch_test(self, observed_counts: Dict[str, int],
                                     expected_ratio: float = 0.5) -> Dict:
        """Test for sample ratio mismatch in A/B test."""
        total = sum(observed_counts.values())
        expected = {k: total * expected_ratio for k in observed_counts}
        observed_arr = np.array(list(observed_counts.values()))
        expected_arr = np.array(list(expected.values()))
        chi2, p_value, dof, expected_vals = chi2_contingency(
            np.array([observed_arr, expected_arr])
        )
        return {
            "chi2_statistic": round(float(chi2), 4),
            "p_value": round(float(p_value), 4),
            "significant": p_value < 0.05,
            "observed": observed_counts,
            "expected": {k: round(v, 0) for k, v in expected.items()},
        }

    def validate_power_assumptions(self, baseline_rate: float, sample_size: int,
                                     effect_size: float) -> Dict:
        """Validate all power assumptions for an A/B test."""
        actual_power = self.calculate_power(sample_size, effect_size)
        mde = self.minimum_detectable_effect(sample_size)
        required_n = self.calculate_sample_size_two_proportions(
            baseline_rate, baseline_rate * (1 - effect_size)
        )
        return {
            "baseline_rate": baseline_rate,
            "effect_size": effect_size,
            "sample_size_per_group": sample_size,
            "actual_power": round(actual_power, 4),
            "power_sufficient": actual_power >= self.power,
            "mde_at_n": mde,
            "required_n_per_group": required_n,
            "sample_sufficient": sample_size >= required_n,
            "recommendation": self._generate_recommendation(actual_power, sample_size, required_n),
        }

    def _generate_recommendation(self, power: float, current_n: int, required_n: int) -> str:
        """Generate recommendation based on power analysis."""
        if power >= self.power and current_n >= required_n:
            return "Sufficient power. Proceed with the test."
        elif power < self.power and current_n < required_n:
            return f"Insufficient power ({power:.1%}). Need {required_n - current_n:,} more samples per group."
        elif power >= self.power:
            return "Power sufficient but sample larger than needed. Consider reducing for efficiency."
        else:
            return f"Power below threshold ({power:.1%} < {self.power:.1%}). Increase sample size."

    def ab_test_sample_size_calculator(self, baseline_rate: float, mde: float,
                                        alpha: float = None, power: float = None) -> Dict:
        """Complete sample size calculator for A/B tests."""
        alpha = alpha or self.significance_level
        power = power or self.power
        treatment_rate = baseline_rate * (1 - mde)
        n = self.calculate_sample_size_two_proportions(baseline_rate, treatment_rate, alpha, power)
        actual_power = self.calculate_power(n, mde)
        return {
            "baseline_rate": baseline_rate,
            "expected_treatment_rate": round(treatment_rate, 4),
            "absolute_effect": round(baseline_rate - treatment_rate, 4),
            "relative_effect_pct": round(mde * 100, 2),
            "sample_size_per_group": n,
            "total_sample_size": n * 2,
            "actual_power": round(actual_power, 4),
            "significance_level": alpha,
            "estimated_duration_days": self._estimate_duration(n),
        }

    def _estimate_duration(self, n_per_group: int) -> int:
        """Estimate test duration in days."""
        daily_rate = 100  # Assumed customers per day
        return int(np.ceil(n_per_group / daily_rate))
