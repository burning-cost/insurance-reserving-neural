# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-reserving-neural: Individual Claims Neural Reserving Demo
# MAGIC
# MAGIC This notebook demonstrates the full workflow:
# MAGIC 1. Generate synthetic claims data (SPLICE-inspired)
# MAGIC 2. Train FNN+ reserver on settled claims
# MAGIC 3. Predict RBNS reserves for open claims
# MAGIC 4. Evaluate with MALE and OCLerr
# MAGIC 5. Compare against chain-ladder baseline
# MAGIC 6. Bootstrap uncertainty quantification (Solvency II percentiles)
# MAGIC
# MAGIC **Architecture note:** FNN+ (feedforward with case estimate features) is the default.
# MAGIC Avanzi et al. (2025) showed case estimates reduce MALE by 15-25% vs plain FNN.
# MAGIC LSTM is available but ~4x slower; use it for portfolios with 10+ year development tails.

# COMMAND ----------

# MAGIC %pip install insurance-reserving-neural[torch] --quiet

# COMMAND ----------

import polars as pl
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from insurance_reserving_neural.synthetic import SyntheticClaimsGenerator
from insurance_reserving_neural.models import FNNReserver
from insurance_reserving_neural.data import prepare_features
from insurance_reserving_neural.metrics import (
    mean_absolute_log_error,
    ocl_error,
    reserve_range,
    chain_ladder_reserve,
)
from insurance_reserving_neural.bootstrap import BootstrapReserver

print("insurance-reserving-neural loaded successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Generate Synthetic Claims Data
# MAGIC
# MAGIC We generate 5,000 individual claims with three types: property, liability, bodily injury.
# MAGIC Each claim is tracked quarterly from accident to settlement.
# MAGIC The panel format has one row per claim per valuation date.

# COMMAND ----------

gen = SyntheticClaimsGenerator(
    n_claims=5_000,
    n_periods=12,         # 3 years of accident quarters
    max_dev_quarters=20,  # up to 5 years development
    inflation_rate=0.07,  # 7% annual claims inflation
    random_state=42,
)
df = gen.generate()

print(f"Total rows (claim-quarter observations): {len(df):,}")
print(f"Unique claims: {df['claim_id'].n_unique():,}")
print(f"\nClaim type distribution:")
print(df.group_by("claim_type").len().sort("len", descending=True))

# COMMAND ----------

# Show what the data looks like for one claim
sample_claim = df["claim_id"][0]
print(f"\nDevelopment history for claim {sample_claim}:")
print(
    df.filter(pl.col("claim_id") == sample_claim)
    .select([
        "valuation_date", "cumulative_paid", "case_estimate",
        "n_case_revisions", "is_open", "ultimate"
    ])
    .sort("valuation_date")
)

# COMMAND ----------

# Split into train/test (split was assigned by the generator — no leakage)
train_df = df.filter(pl.col("split") == "train")
test_df  = df.filter(pl.col("split") == "test")

print(f"Training set: {train_df['claim_id'].n_unique():,} claims, {len(train_df):,} observations")
print(f"Test set:     {test_df['claim_id'].n_unique():,} claims,  {len(test_df):,} observations")

# Training uses only settled claims
n_settled_train = (train_df["is_open"] == False).sum()
print(f"Settled training observations: {n_settled_train:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Train FNN+ Reserver
# MAGIC
# MAGIC The FNN+ model extracts:
# MAGIC - Payment features: log(cumulative_paid), development quarter, calendar quarter
# MAGIC - Case estimate summaries: n_revisions, mean, std, trend, largest revision, prop upward
# MAGIC - Claim covariates: claim_type (int), litigation flag, fault proportion
# MAGIC
# MAGIC Training uses only settled claims (known ultimate). Log-space MSE loss with gradient clipping.

# COMMAND ----------

model = FNNReserver(
    use_case_estimates=True,   # FNN+ includes case estimate history features
    hidden_sizes=(64, 32, 16), # three hidden layers
    dropout=0.1,
    learning_rate=1e-3,
    weight_decay=1e-4,
    max_epochs=100,
    patience=10,
    batch_size=512,
    grad_clip=1.0,
    random_state=42,
    verbose=True,
)

model.fit(train_df)

# COMMAND ----------

# Training curve
history = model.training_history_
print(f"Trained for {len(history)} epochs")
print(f"First epoch — train_loss: {history[0]['train_loss']:.4f}, val_loss: {history[0]['val_loss']:.4f}")
print(f"Final epoch — train_loss: {history[-1]['train_loss']:.4f}, val_loss: {history[-1]['val_loss']:.4f}")
print(f"Improvement: {(1 - history[-1]['val_loss'] / history[0]['val_loss']) * 100:.1f}%")
print(f"Duan smearing factor: {model._smearing_factor:.4f}")
print(f"Feature count: {len(model._feature_cols)}")
print(f"Features: {model._feature_cols}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Predict RBNS Reserves

# COMMAND ----------

# Predict on test set (open and settled claims)
preds = model.predict(test_df)

# RBNS reserve = sum of predicted outstanding for open claims only
open_preds = preds.filter(pl.col("is_open") == True)
rbns_reserve = float(open_preds["predicted_outstanding"].sum())
print(f"RBNS Reserve (open claims): £{rbns_reserve:,.0f}")

# Distribution of per-claim predictions
print(f"\nPer-claim predicted outstanding:")
print(f"  Median: £{float(open_preds['predicted_outstanding'].median()):,.0f}")
print(f"  Mean:   £{float(open_preds['predicted_outstanding'].mean()):,.0f}")
print(f"  P95:    £{float(np.percentile(open_preds['predicted_outstanding'].to_numpy(), 95)):,.0f}")
print(f"  Max:    £{float(open_preds['predicted_outstanding'].max()):,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Evaluate on Settled Claims

# COMMAND ----------

# Evaluate on test set settled claims (known ultimate available)
settled_test = test_df.filter(
    (pl.col("is_open") == False)
    & (pl.col("ultimate").is_not_null())
    & ((pl.col("ultimate") - pl.col("cumulative_paid")) > 1.0)
)
preds_settled = model.predict(settled_test)

male = mean_absolute_log_error(preds_settled)
oclerr = ocl_error(preds_settled)

print(f"MALE (Mean Absolute Log Error): {male:.4f}")
print(f"OCLerr (portfolio-level bias):  {oclerr:+.4f}  ({oclerr*100:+.1f}%)")
print()
print("MALE interpretation:")
print(f"  0.30 = reasonable first model")
print(f"  0.25 = good (Avanzi et al. published range on SPLICE data)")
print(f"  Our result: {male:.4f}")
print()
if oclerr > 0:
    print(f"OCLerr: model OVER-reserves by {abs(oclerr)*100:.1f}% at portfolio level")
else:
    print(f"OCLerr: model UNDER-reserves by {abs(oclerr)*100:.1f}% at portfolio level (regulatory concern)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Baseline: FNN without Case Estimates
# MAGIC
# MAGIC Reproduce the FNN vs FNN+ comparison from Avanzi et al. (2025).

# COMMAND ----------

model_plain = FNNReserver(
    use_case_estimates=False,  # plain FNN — no case estimate history
    hidden_sizes=(64, 32, 16),
    max_epochs=100,
    patience=10,
    random_state=42,
    verbose=False,
)
model_plain.fit(train_df)
preds_plain = model_plain.predict(settled_test)

male_plain = mean_absolute_log_error(preds_plain)
male_plus  = male

print(f"Plain FNN MALE:  {male_plain:.4f}")
print(f"FNN+ MALE:       {male_plus:.4f}")
print(f"Improvement:     {(1 - male_plus/male_plain)*100:.1f}%")
print()
print("Case estimates help when improvement > 0 — consistent with Avanzi et al. finding")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Chain-Ladder Benchmark
# MAGIC
# MAGIC PRA SS8/24 requires benchmarking neural models against traditional methods.
# MAGIC We compute the chain-ladder reserve directly from the individual claims data.

# COMMAND ----------

df_with_features = prepare_features(df, use_case_estimates=False)
cl_result = chain_ladder_reserve(df_with_features)

print(f"Chain-ladder reserve: £{cl_result['reserve']:,.0f}")
print(f"Neural RBNS reserve:  £{rbns_reserve:,.0f}")
print(f"Ratio (neural/CL):    {rbns_reserve / cl_result['reserve']:.3f}")
print()
print(f"Development factors (first 8 quarters):")
for i, f in enumerate(cl_result['dev_factors'][:8]):
    print(f"  q{i} -> q{i+1}: {f:.4f}")
print(f"Tail CDF at quarter 0: {cl_result['cdfs'][0]:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Bootstrap Uncertainty (Solvency II Percentiles)
# MAGIC
# MAGIC Solvency II requires the reserve distribution, not just a point estimate.
# MAGIC PRA expects P75 (best estimate + risk margin) and P99.5 (SCR).

# COMMAND ----------

boot = BootstrapReserver(
    base_model=FNNReserver(
        use_case_estimates=True,
        max_epochs=50,  # faster for demo; use 100+ in production
        random_state=42,
    ),
    n_boot=500,  # 2000+ for production Solvency II reporting
    random_state=42,
    verbose=True,
)

boot.fit(train_df)

# Generate reserve distribution for open claims
open_test = test_df.filter(pl.col("is_open") == True)
dist = boot.reserve_distribution(open_test)

print(f"\nRBNS Reserve Distribution:")
print(f"  Point estimate: £{dist['point_estimate']:,.0f}")
print(f"  Bootstrap mean: £{dist['mean']:,.0f}")
print(f"  Bootstrap std:  £{dist['std']:,.0f}")
print(f"  CoV:            {dist['coeff_of_variation']:.3f}")
print()
print(f"Solvency II percentiles:")
print(f"  P10:  £{dist['P10']:,.0f}")
print(f"  P50:  £{dist['P50']:,.0f}  (median)")
print(f"  P75:  £{dist['P75']:,.0f}  (typical risk margin basis)")
print(f"  P90:  £{dist['P90']:,.0f}")
print(f"  P99_5: £{dist['P99_5']:,.0f}  (SCR basis)")
print()
risk_margin = dist['P75'] - dist['P50']
print(f"Indicative risk margin (P75-P50): £{risk_margin:,.0f}")

# COMMAND ----------

# Residual diagnostics
resid = boot.residual_summary()
print("Training residual diagnostics:")
print(f"  Count:            {resid['n_residuals']:,}")
print(f"  Mean:             {resid['mean']:.4f}  (near-zero = unbiased)")
print(f"  Std:              {resid['std']:.4f}")
print(f"  Skewness:         {resid['skewness']:.3f}  (>0 = right-skew expected for insurance)")
print(f"  Kurtosis:         {resid['kurtosis']:.3f}  (>0 = fat tails)")
print(f"  Normal at 5%:     {resid['is_normal_at_5pct']}  (expect False for insurance data)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. LSTM Reserver (Optional — Longer Training)
# MAGIC
# MAGIC Processes the full sequence of payments and case estimates per claim.
# MAGIC For most UK portfolios FNN+ is sufficient. Use LSTM for portfolios with
# MAGIC long development tails where the revision sequence history matters.

# COMMAND ----------

from insurance_reserving_neural.models import LSTMReserver

lstm_model = LSTMReserver(
    hidden_size=32,
    num_layers=2,
    max_epochs=30,
    patience=5,
    batch_size=64,
    random_state=42,
    verbose=True,
)

lstm_model.fit(train_df)

lstm_reserve = lstm_model.reserve(test_df)
print(f"\nLSTM RBNS Reserve: £{lstm_reserve:,.0f}")
print(f"FNN+ RBNS Reserve: £{rbns_reserve:,.0f}")
print(f"Ratio:             {lstm_reserve/rbns_reserve:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Metric | Value |
# MAGIC |--------|-------|
# MAGIC | FNN+ MALE | - |
# MAGIC | Plain FNN MALE | - |
# MAGIC | OCLerr | - |
# MAGIC | RBNS Reserve (FNN+) | - |
# MAGIC | CL Reserve | - |
# MAGIC | P99.5 Reserve | - |
# MAGIC
# MAGIC **Key findings:**
# MAGIC - Case estimate features (FNN+) provide meaningful MALE improvement over plain FNN
# MAGIC - OCLerr should be small — the model is approximately unbiased at portfolio level
# MAGIC - Bootstrap P99.5 provides the SCR-level reserve estimate required by Solvency II
# MAGIC - Chain-ladder comparison is built-in for PRA SS8/24 model validation documentation
