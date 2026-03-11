# insurance-reserving-neural

Individual neural claims reserving for UK insurance teams. Per-claim RBNS reserve estimates with bootstrap uncertainty, benchmarked against chain-ladder.

## The problem

Chain-ladder treats your whole portfolio as one blended triangle. A bodily injury claim that has been through two failed settlement offers and three case estimate revisions gets the same development factor as a straightforward property claim that settled in six months. The method discards everything your claims handlers actually know.

The PRA's 2023 thematic review found this directly: insurers using aggregate triangle methods were systematically failing to capture heterogeneous claims inflation. Bodily injury running at 15% and property at 3% — blended triangles under-reserve one and over-reserve the other.

Individual neural reserving predicts each claim's outstanding liability using its own payment history, case estimate revision pattern, claim type, litigation status, and development stage. Sum the predictions over open claims to get the RBNS reserve. The approach has been in the academic literature since 2022 and has demonstrably better performance metrics than chain-ladder on every dataset tested.

There was no Python library for it. Until now.

## What this library does

- Generates realistic synthetic claims data (SPLICE-inspired) for testing and onboarding
- Extracts per-claim features from panel data: payment summaries, case estimate history, development timing
- Trains feedforward neural networks (FNN+) and LSTMs on individual claim development histories
- Applies Duan (1983) smearing bias correction to log-space predictions
- Produces per-claim outstanding predictions and aggregate portfolio RBNS reserves
- Quantifies reserve uncertainty via residual bootstrap (P10/P50/P75/P90/P99.5 for Solvency II)
- Computes an aggregate chain-ladder reserve from the same data for regulatory benchmarking

## Installation

```bash
pip install insurance-reserving-neural           # core (no PyTorch)
pip install insurance-reserving-neural[torch]    # with PyTorch for neural models
```

## Quick start

```python
import polars as pl
from insurance_reserving_neural.synthetic import SyntheticClaimsGenerator
from insurance_reserving_neural.models import FNNReserver
from insurance_reserving_neural.metrics import mean_absolute_log_error, ocl_error, chain_ladder_reserve
from insurance_reserving_neural.bootstrap import BootstrapReserver

# Generate synthetic data — replace with your own panel DataFrame
gen = SyntheticClaimsGenerator(n_claims=5000, random_state=42)
df = gen.generate()

train_df = df.filter(pl.col("split") == "train")
test_df  = df.filter(pl.col("split") == "test")

# Train FNN+ (feedforward with case estimate features)
model = FNNReserver(use_case_estimates=True, max_epochs=100)
model.fit(train_df)

# Per-claim predictions
preds = model.predict(test_df)
print(f"MALE:    {mean_absolute_log_error(preds):.4f}")
print(f"OCLerr:  {ocl_error(preds):.4f}")

# Portfolio RBNS reserve
reserve = model.reserve(test_df.filter(pl.col("is_open") == True))
print(f"RBNS reserve: £{reserve:,.0f}")

# Chain-ladder benchmark
from insurance_reserving_neural.data import prepare_features
cl_result = chain_ladder_reserve(prepare_features(df))
print(f"Chain-ladder reserve: £{cl_result['reserve']:,.0f}")

# Bootstrap uncertainty (Solvency II percentiles)
boot = BootstrapReserver(model, n_boot=1000)
boot.fit(train_df)
dist = boot.reserve_distribution(test_df.filter(pl.col("is_open") == True))
print(f"P50: £{dist['P50']:,.0f}  P90: £{dist['P90']:,.0f}  P99.5: £{dist['P99_5']:,.0f}")
```

## Data format

Your data needs to be in panel format: one row per claim per valuation date. Each row contains the claim's history as of that date.

**Required columns:**

| Column | Type | Description |
|--------|------|-------------|
| `claim_id` | str | Unique claim identifier |
| `accident_date` | date | Date of loss |
| `valuation_date` | date | As-at date for this observation |
| `cumulative_paid` | float | Total payments to valuation_date |
| `case_estimate` | float | Current case reserve at valuation_date |
| `is_open` | bool | True if claim still open |
| `ultimate` | float | Final settlement amount (null for open claims) |

**Case estimate history columns** (needed for FNN+, the recommended model):

| Column | Description |
|--------|-------------|
| `n_case_revisions` | Number of case estimate changes to date |
| `case_estimate_mean` | Mean case estimate over revision history |
| `case_estimate_std` | Std dev of revision history |
| `case_estimate_trend` | Linear trend of case estimates |
| `largest_case_revision` | Largest single absolute revision |
| `prop_upward_revisions` | Fraction of revisions that were upward |

**User covariates:** any columns starting with `feat_` are treated as model inputs. Standard examples: `feat_claim_type` (int), `feat_litigation` (0/1), `feat_fault` (float).

## Architecture

**FNN+ is the default.** Avanzi, Lambrianidis, Taylor, Wong (2025) tested four architectures on SPLICE-simulated data and found that case estimate summaries reduce MALE by 15–25% compared to payment history alone. The LSTM adds only marginal improvement over FNN+ at 4–8× the training cost. For UK portfolios with moderate development tails, FNN+ is the right default.

```
Model architecture:
Input (features + case estimate summaries)
  → Linear(n_features, 64) → BatchNorm → ReLU → Dropout(0.1)
  → Linear(64, 32) → BatchNorm → ReLU → Dropout(0.1)
  → Linear(32, 16) → BatchNorm → ReLU → Dropout(0.1)
  → Linear(16, 1)
  → log-outstanding prediction
  → exp(ŷ) × Duan smearing factor → outstanding prediction
```

**Loss function:** MSE on log(outstanding). Log-space training is not optional — raw MSE on payment amounts is dominated by the top 1% of large claims and produces unstable gradients.

**Bias correction:** Duan (1983) smearing estimator. `E[exp(Y)] ≠ exp(E[Y])` — naive back-transformation understates the mean. The smearing factor `b = mean(exp(residuals))` corrects this.

## Metrics

**MALE** (Mean Absolute Log Error): per-claim accuracy in log space. Lower is better. 0.3 is reasonable for a first model; published results on SPLICE data show FNN+ achieving 0.25–0.35.

**OCLerr** (Outstanding Claims Liability Error): portfolio-level signed bias. `(predicted_total - actual_total) / actual_total`. What the finance director looks at. Under-reserving (negative) is the regulatory risk.

**Chain-ladder reserve:** aggregate benchmark computed from the same individual data. Required by PRA SS8/24 for model validation documentation.

## Regulatory context

PRA SS8/24 (effective December 2024) states that relying solely on historical triangle extrapolation "is unlikely to satisfy the Directive requirement for a probability-weighted average of future cash-flows." This library provides individual-data methods that directly address this requirement.

The built-in chain-ladder comparison and bootstrap uncertainty quantification produce the evidence an internal model validation team needs for sign-off.

## References

- Avanzi, Lambrianidis, Taylor, Wong (2025). *On the use of case estimate and transactional payment data in neural networks for individual loss reserving.* arXiv:2601.05274
- Richman, Wüthrich (2026). *From Chain-Ladder to Individual Claims Reserving.* arXiv:2602.15385
- Schneider, Schwab (2025). *Advancing Loss Reserving: A Hybrid Neural Network Approach.* JRI / SSRN:4769020
- Duan N (1983). *Smearing estimate: A nonparametric retransformation method.* JASA 78(383), 605–610
- PRA SS8/24: Solvency II Calculation of Technical Provisions (November 2024)

## License

MIT. Built by [Burning Cost](https://github.com/burning-cost).
