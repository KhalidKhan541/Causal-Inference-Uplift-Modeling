# Causal Inference & Uplift Modeling Framework

A production-grade framework for estimating individual treatment effects and optimizing campaign targeting using causal inference and uplift modeling techniques.

## Overview

Traditional A/B testing tells you *what happened* — this framework tells you *who to target*. By combining propensity score methods with uplift modeling, it identifies the customers most likely to respond positively to a treatment (e.g., discount, ad exposure, email) while filtering out those who would have converted anyway.

## Architecture

```
causal_framework/
├── preprocessing.py       # Data loading, validation, feature engineering
├── data_utils.py          # Synthetic data generation, distribution utilities
├── propensity.py          # Propensity score estimation (Logistic, GBM, CB)
├── causal_inference.py    # ATE, ATT, CATE estimation (IPW, AIPW, TMLE, DR-Learner)
├── uplift_models.py       # Uplift modeling (T-Learner, S-Learner, X-Learner, R-Learner, Causal Forest)
├── uplift_evaluation.py   # Uplift metrics, Qini curves, AUUC, policy evaluation
├── power_analysis.py      # Sample size and power calculations
├── reporting.py           # HTML reports, visualizations, summary tables
├── pipeline.py            # End-to-end orchestration
├── run.py                 # CLI entry point
└── configs/
    └── default.yaml       # Default configuration
```

## Features

### Propensity Score Estimation
- Logistic Regression, Gradient Boosting, CatBoost
- Propensity score distribution diagnostics
- Common support / overlap checks
- Covariate balance assessment (SMD)

### Causal Inference Methods
| Method | Type | Description |
|--------|------|-------------|
| IPW | ATE/ATT | Inverse Probability Weighting |
| AIPW | ATE | Augmented IPW (doubly robust) |
| TMLE | ATE | Targeted Maximum Likelihood Estimation |
| DR-Learner | CATE | Doubly Robust Learner |
| Causal Forest | CATE | Generalized Random Forest (GRF) |

### Uplift Modeling
- **T-Learner**: Separate models for treatment/control
- **S-Learner**: Single model with treatment as feature
- **X-Learner**: Cross-learner with propensity weighting
- **R-Learner**: Robinson-transformed residualization
- **Causal Forest**: Non-parametric heterogeneous treatment effects

### Evaluation Metrics
- Qini coefficient and Qini curves
- Area Under Uplift Curve (AUUC)
- Net incremental revenue at top-k%
- Policy value simulation
- Feature importance for treatment effect heterogeneity

### Power Analysis
- Sample size calculations for ATE detection
- Minimum detectable effect (MDE) computation
- Power curves across effect sizes and sample sizes

## Quick Start

```bash
pip install -r requirements.txt

# Run full pipeline with synthetic data
python -m causal_framework.run --config configs/default.yaml --output results/

# Run on your own data
python -m causal_framework.run --data your_data.csv --outcome outcome_col --treatment treatment_col --features f1,f2,f3
```

## Configuration

Edit `configs/default.yaml`:

```yaml
data:
  path: null            # null = synthetic data
  outcome_col: converted
  treatment_col: treated
  feature_cols: [age, income, history, recency, channel]
  test_size: 0.2
  random_state: 42

propensity:
  method: gradient_boosting
  n_estimators: 200

inference:
  methods: [ipw, aipw, dr_learner]
  n_bootstrap: 1000

uplift:
  models: [t_learner, s_learner, x_learner, causal_forest]

power:
  alpha: 0.05
  power: 0.80
  mde: 0.02
```

## Output

The pipeline generates:
- `report.html` — Interactive HTML report with all visualizations
- `ate_estimates.csv` — Average treatment effect estimates per method
- `cate_estimates.csv` — Individual-level conditional treatment effects
- `uplift_scores.csv` — Uplift scores for each customer
- `qini_curves.png` — Model comparison plot
- `propensity_diagnostics.png` — Propensity score distribution and balance

## Example Results

```
Method       ATE Estimate   95% CI              p-value
─────────────────────────────────────────────────────────
IPW          0.087          [0.072, 0.102]      <0.001
AIPW         0.083          [0.070, 0.096]      <0.001
TMLE         0.084          [0.071, 0.097]      <0.001
DR-Learner   0.085          [0.069, 0.101]      <0.001
Causal Forest 0.082         [0.065, 0.099]      <0.001

Uplift Model   Qini    AUUC    Top-20% Lift
──────────────────────────────────────────────
T-Learner      0.12    0.09    +34%
X-Learner      0.15    0.11    +41%
Causal Forest  0.18    0.13    +48%
```

## Dependencies

- numpy, pandas, scipy
- scikit-learn, xgboost, lightgbm, catboost
- causalml, econml (optional, for advanced estimators)
- matplotlib, seaborn
- pyyaml
- statsmodels
