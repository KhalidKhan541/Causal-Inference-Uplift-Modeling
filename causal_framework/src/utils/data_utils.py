import numpy as np
import pandas as pd
from typing import Dict, Any
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_sample_data(n_customers: int = 5000) -> pd.DataFrame:
    """Generate synthetic marketing intervention data for testing."""
    np.random.seed(42)
    age = np.random.randint(18, 70, n_customers)
    income = np.random.lognormal(10.5, 0.8, n_customers).astype(int)
    tenure = np.random.randint(1, 60, n_customers)
    gender = np.random.choice(["Male", "Female"], n_customers)
    region = np.random.choice(["North", "South", "East", "West"], n_customers)
    segment = np.random.choice(["Premium", "Standard", "Basic"], n_customers, p=[0.2, 0.5, 0.3])
    channel = np.random.choice(["Organic", "Paid", "Referral", "Email"], n_customers)
    contract = np.random.choice(["Monthly", "Annual", "None"], n_customers, p=[0.4, 0.3, 0.3])
    # Derived features
    purchase_freq = np.random.poisson(5, n_customers) + tenure * 0.1
    avg_order = np.random.lognormal(4, 0.5, n_customers)
    total_spend = purchase_freq * avg_order
    support_tickets = np.random.poisson(1.5, n_customers)
    sessions = np.random.randint(5, 100, n_customers)
    days_since = np.random.randint(1, 90, n_customers)
    # Treatment assignment (propensity depends on features)
    propensity = 1 / (1 + np.exp(-(
        0.02 * (age - 35) / 15 +
        0.01 * (income - 50000) / 30000 +
        0.005 * tenure +
        0.03 * (segment == "Premium").astype(int) -
        0.02 * (segment == "Basic").astype(int)
    )))
    treatment = np.random.binomial(1, propensity)
    # Outcome (depends on treatment + confounders)
    true_ate = 0.05
    baseline = 1 / (1 + np.exp(-(
        -2.0 +
        0.01 * (age - 35) / 15 +
        0.008 * (income - 50000) / 30000 +
        0.005 * tenure +
        0.1 * (segment == "Premium").astype(int) -
        0.05 * support_tickets / 3 +
        true_ate * treatment +
        np.random.normal(0, 0.3, n_customers)
    )))
    outcome = np.random.binomial(1, baseline)
    return pd.DataFrame({
        "customer_id": [f"CUST_{i:05d}" for i in range(n_customers)],
        "date": pd.date_range("2024-01-01", periods=n_customers, freq="D"),
        "age": age, "income": income, "tenure_months": tenure,
        "gender": gender, "region": region, "customer_segment": segment,
        "acquisition_channel": channel, "contract_type": contract,
        "purchase_frequency": purchase_freq, "avg_order_value": np.round(avg_order, 2),
        "total_spend": np.round(total_spend, 2), "support_tickets": support_tickets,
        "session_count": sessions, "days_since_last_purchase": days_since,
        "treatment": treatment, "outcome": outcome,
    })


def calculate_ate(y_true: np.ndarray, treatment: np.ndarray) -> Dict[str, float]:
    """Calculate Average Treatment Effect."""
    treated = y_true[treatment == 1]
    control = y_true[treatment == 0]
    ate = float(np.mean(treated) - np.mean(control))
    se = float(np.sqrt(np.var(treated) / len(treated) + np.var(control) / len(control)))
    ci_lower = ate - 1.96 * se
    ci_upper = ate + 1.96 * se
    # T-test
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(treated, control)
    return {
        "ate": round(ate, 6),
        "se": round(se, 6),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "significant": p_value < 0.05,
        "n_treated": int(len(treated)),
        "n_control": int(len(control)),
        "mean_treated": round(float(np.mean(treated)), 4),
        "mean_control": round(float(np.mean(control)), 4),
    }
