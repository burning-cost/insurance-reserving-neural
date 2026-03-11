"""
synthetic.py — SPLICE-inspired synthetic individual claims generator.

Generates realistic claim trajectories with partial payments, case estimate
revisions, reopenings, and settlement. The generator is non-negotiable for
two reasons: (1) tests cannot use proprietary insurer data, and (2) new users
need to run a realistic end-to-end workflow before connecting their own data.

The generator is inspired by the SPLICE framework (Avanzi, Taylor, Wang, Ho 2021)
but implemented in pure Python/NumPy/Polars without the R dependency. We
reproduce the key behavioural features:
  - Claims reported with a delay from accident to reporting date
  - Partial payments at irregular intervals until settlement
  - Case estimates set at reporting and revised upward or downward over time
  - Settlement timing follows a parametric survival model
  - Optional reopenings after settlement (with corrected trajectory mechanics)
  - Three claim types with distinct severity and development profiles

Design choices
--------------
We generate claims up to a configurable valuation date, then slice each claim's
history at each quarter to produce the panel dataset format. This means
n_claims * average_dev_periods rows in the output.

The log-normal severity with mixture is chosen because:
  - It produces the heavy right tail that makes reserving hard
  - The claim type mixture creates heterogeneous inflation profiles
  - It matches what SPLICE and SynthETIC use in their simulations

Key invariants maintained throughout:
  - cumulative_paid is non-decreasing within a claim
  - cumulative_paid <= ultimate at every development quarter
  - For settled claims at last observation: cumulative_paid == ultimate
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import polars as pl
from datetime import date


@dataclass
class SeverityConfig:
    """
    Parameters for the log-normal severity distribution of a claim type.

    log_mean and log_std are the mean and std of log(ultimate),
    not the mean and std of ultimate itself.
    """
    log_mean: float = 8.0
    log_std: float = 1.2
    settlement_rate: float = 0.3
    payment_frequency: float = 0.6
    case_revision_freq: float = 0.4
    large_loss_threshold: float = 50_000.0
    large_loss_prob: float = 0.05
    large_log_mean: float = 11.0
    large_log_std: float = 0.8


CLAIM_TYPE_CONFIGS = {
    "property": SeverityConfig(
        log_mean=7.0, log_std=1.0, settlement_rate=0.45,
        payment_frequency=0.5, case_revision_freq=0.25,
        large_loss_threshold=20_000.0, large_loss_prob=0.03,
        large_log_mean=9.5, large_log_std=0.6,
    ),
    "liability": SeverityConfig(
        log_mean=8.5, log_std=1.4, settlement_rate=0.20,
        payment_frequency=0.55, case_revision_freq=0.50,
        large_loss_threshold=100_000.0, large_loss_prob=0.08,
        large_log_mean=12.0, large_log_std=0.9,
    ),
    "bodily_injury": SeverityConfig(
        log_mean=9.0, log_std=1.6, settlement_rate=0.15,
        payment_frequency=0.60, case_revision_freq=0.60,
        large_loss_threshold=200_000.0, large_loss_prob=0.12,
        large_log_mean=13.0, large_log_std=1.0,
    ),
}

CLAIM_TYPE_NAMES = list(CLAIM_TYPE_CONFIGS.keys())
CLAIM_TYPE_PROBS = [0.40, 0.35, 0.25]  # property, liability, BI


class SyntheticClaimsGenerator:
    """
    Generate a synthetic individual claims panel for testing and demonstration.

    The output is a Polars DataFrame in the panel format expected by
    prepare_features() and the model classes. Each row is one claim at one
    valuation date.

    Parameters
    ----------
    n_claims : int
        Number of claims to generate. 5,000+ is needed for stable FNN training.
        Default 3,000 is sufficient for unit tests.
    n_periods : int
        Number of accident quarters to spread claims across. Default 12 (3 years).
    max_dev_quarters : int
        Maximum development quarters tracked per claim. Default 20 (5 years).
    settlement_rate : float | None
        Override the per-claim-type settlement rate with a single value.
    severity_distribution : str
        'lognormal' (default) or 'pareto'. Pareto gives heavier tail.
    inflation_rate : float
        Annual claims inflation rate applied to payments. Default 0.07 (7%).
    reopen_prob : float
        Probability a settled claim reopens. Default 0.05.
    random_state : int | None
        Random seed for reproducibility.
    train_frac : float
        Fraction of claims in the training split. Default 0.7.
    """

    def __init__(
        self,
        n_claims: int = 3_000,
        n_periods: int = 12,
        max_dev_quarters: int = 20,
        settlement_rate: Optional[float] = None,
        severity_distribution: str = "lognormal",
        inflation_rate: float = 0.07,
        reopen_prob: float = 0.05,
        random_state: Optional[int] = None,
        train_frac: float = 0.70,
    ) -> None:
        self.n_claims = n_claims
        self.n_periods = n_periods
        self.max_dev_quarters = max_dev_quarters
        self.settlement_rate = settlement_rate
        self.severity_distribution = severity_distribution
        self.inflation_rate = inflation_rate
        self.reopen_prob = reopen_prob
        self.random_state = random_state
        self.train_frac = train_frac

        self._rng = np.random.default_rng(random_state)

    def generate(self) -> pl.DataFrame:
        """
        Generate the full claims panel DataFrame.

        Returns
        -------
        pl.DataFrame
            Panel with one row per claim per valuation quarter. Columns include
            all required schema columns plus case estimate history summaries,
            feat_* covariates, and a train/test split indicator.
        """
        claims = self._generate_claims()
        rows = self._expand_to_panel(claims)
        df = pl.DataFrame(rows)
        df = self._add_split(df)
        return df

    def _generate_claims(self) -> list[dict]:
        base_date = date(2020, 1, 1)
        claims = []

        for i in range(self.n_claims):
            claim_type = self._rng.choice(CLAIM_TYPE_NAMES, p=CLAIM_TYPE_PROBS)
            cfg = CLAIM_TYPE_CONFIGS[claim_type]
            eff_settlement_rate = (
                self.settlement_rate if self.settlement_rate is not None
                else cfg.settlement_rate
            )

            accident_quarter = int(self._rng.integers(0, self.n_periods))
            accident_date = self._quarter_to_date(base_date, accident_quarter)

            if claim_type == "bodily_injury":
                report_delay_quarters = int(self._rng.integers(1, 4))
            else:
                report_delay_quarters = int(self._rng.integers(0, 2))

            reporting_date = self._quarter_to_date(
                base_date, accident_quarter + report_delay_quarters
            )

            # Draw initial ultimate from log-normal (possibly large loss)
            is_large = self._rng.random() < cfg.large_loss_prob
            if is_large:
                if self.severity_distribution == "pareto":
                    alpha, x_m = 2.0, np.exp(cfg.large_log_mean - 1.0 / 2.0)
                    initial_ultimate = float(x_m * (1.0 / self._rng.random()) ** (1.0 / alpha))
                else:
                    initial_ultimate = float(self._rng.lognormal(cfg.large_log_mean, cfg.large_log_std))
            else:
                if self.severity_distribution == "pareto":
                    alpha, x_m = 3.0, np.exp(cfg.log_mean - 1.0 / 3.0)
                    initial_ultimate = float(x_m * (1.0 / self._rng.random()) ** (1.0 / alpha))
                else:
                    initial_ultimate = float(self._rng.lognormal(cfg.log_mean, cfg.log_std))

            litigation = int(self._rng.random() < (0.40 if claim_type == "bodily_injury" else 0.15))
            fault = float(np.clip(self._rng.beta(2, 3), 0.0, 1.0))

            trajectory, final_ultimate = self._simulate_trajectory(
                initial_ultimate=initial_ultimate,
                cfg=cfg,
                eff_settlement_rate=eff_settlement_rate,
                accident_quarter=accident_quarter,
                report_delay_quarters=report_delay_quarters,
                base_date=base_date,
            )

            claims.append({
                "claim_id": f"CLM{i:07d}",
                "accident_date": accident_date,
                "reporting_date": reporting_date,
                "claim_type": claim_type,
                "ultimate": final_ultimate,
                "litigation": litigation,
                "fault": fault,
                "trajectory": trajectory,
            })

        return claims

    def _simulate_trajectory(
        self,
        initial_ultimate: float,
        cfg: SeverityConfig,
        eff_settlement_rate: float,
        accident_quarter: int,
        report_delay_quarters: int,
        base_date: date,
    ) -> tuple[list[dict], float]:
        """
        Simulate quarter-by-quarter development for a single claim.

        Invariants maintained:
        - cumulative_paid is non-decreasing at every step
        - cumulative_paid <= current_ultimate at every step
        - At settlement: cumulative_paid == current_ultimate
        - Reopening: current_ultimate increases; cumulative_paid stays non-decreasing

        Returns
        -------
        rows : list[dict]
            One dict per development quarter. Rows store the local state;
            the 'ultimate' field reflects the current_ultimate at THAT point,
            which may differ if reopening occurs.
        final_ultimate : float
            The ultimate at the end of development (used for claim-level storage).
        """
        first_dev = accident_quarter + report_delay_quarters
        cumulative_paid = 0.0
        current_ultimate = initial_ultimate

        initial_ce_factor = self._rng.lognormal(0.0, 0.3)
        initial_case_estimate = float(current_ultimate * initial_ce_factor)
        case_estimates_history = [initial_case_estimate]

        is_open = True
        # Incremental per-quarter inflation factor
        quarterly_inflation = (1.0 + self.inflation_rate) ** 0.25

        rows = []

        for dev_q in range(self.max_dev_quarters):
            abs_quarter = first_dev + dev_q
            val_date = self._quarter_to_date(base_date, abs_quarter)

            # Payment this quarter (only if open and something remains)
            remaining = current_ultimate - cumulative_paid
            if is_open and remaining > 0 and self._rng.random() < cfg.payment_frequency:
                # Fraction of remaining — beta distributed (right-skewed, most payments small)
                pay_frac = self._rng.beta(1.5, 4.0)
                # Apply incremental inflation boost to this quarter's payment
                payment = float(remaining * pay_frac * quarterly_inflation)
                # Hard cap: never exceed remaining
                payment = min(payment, remaining)
                cumulative_paid = cumulative_paid + payment
                # Guarantee invariant
                cumulative_paid = min(cumulative_paid, current_ultimate)

            # Case estimate revision
            current_ce = case_estimates_history[-1]
            if is_open and self._rng.random() < cfg.case_revision_freq:
                remaining_now = max(0.0, current_ultimate - cumulative_paid)
                revision_factor = self._rng.lognormal(0.0, 0.25)
                new_ce = float(remaining_now * revision_factor)
                case_estimates_history.append(new_ce)
            else:
                case_estimates_history.append(current_ce)

            # Summary statistics for case estimate history
            ces = np.array(case_estimates_history)
            revisions = np.diff(ces)
            n_revisions = len(revisions)

            if len(ces) >= 2 and np.std(ces) > 0:
                x = np.arange(len(ces), dtype=float)
                ce_trend = float(np.polyfit(x, ces, 1)[0])
            else:
                ce_trend = 0.0

            upward = float(np.sum(revisions > 0)) / max(1, n_revisions)

            # Settlement: does claim close this quarter?
            settle_this_quarter = is_open and self._rng.random() < eff_settlement_rate
            if settle_this_quarter:
                # Final payment brings paid to ultimate
                cumulative_paid = current_ultimate
                is_open = False

            rows.append({
                "valuation_date": val_date,
                "cumulative_paid": float(cumulative_paid),
                "case_estimate": float(case_estimates_history[-1]),
                "is_open": bool(is_open),
                "row_ultimate": float(current_ultimate),  # local ultimate for this row
                "n_case_revisions": n_revisions,
                "case_estimate_mean": float(np.mean(ces)),
                "case_estimate_std": float(np.std(ces)) if len(ces) > 1 else 0.0,
                "case_estimate_trend": ce_trend,
                "largest_case_revision": float(np.max(np.abs(revisions))) if len(revisions) > 0 else 0.0,
                "prop_upward_revisions": upward,
            })

            # Possible reopening after settlement (IBNR re-emergence)
            # Only reopen if we just settled and there are more quarters available
            if (not is_open
                    and dev_q < self.max_dev_quarters - 2
                    and self._rng.random() < self.reopen_prob):
                # Extra liability discovered: add to ultimate (never reduce paid)
                extra = float(current_ultimate * self._rng.beta(0.5, 5.0) * 0.1)
                current_ultimate += extra
                is_open = True
                # Update last row to reflect reopening state
                rows[-1]["is_open"] = True
                rows[-1]["row_ultimate"] = float(current_ultimate)

        return rows, current_ultimate

    def _expand_to_panel(self, claims: list[dict]) -> list[dict]:
        """
        Flatten claim trajectories into panel rows.

        We use the per-row ultimate (row_ultimate) to populate the
        ultimate column, so that at every point in time cumulative_paid <= ultimate.
        """
        panel = []
        for claim in claims:
            for row in claim["trajectory"]:
                panel.append({
                    "claim_id": claim["claim_id"],
                    "accident_date": claim["accident_date"],
                    "valuation_date": row["valuation_date"],
                    "claim_type": claim["claim_type"],
                    "cumulative_paid": row["cumulative_paid"],
                    "case_estimate": row["case_estimate"],
                    "is_open": row["is_open"],
                    # Use the row-level ultimate (reflects any reopening at that point)
                    "ultimate": row["row_ultimate"],
                    "n_case_revisions": row["n_case_revisions"],
                    "case_estimate_mean": row["case_estimate_mean"],
                    "case_estimate_std": row["case_estimate_std"],
                    "case_estimate_trend": row["case_estimate_trend"],
                    "largest_case_revision": row["largest_case_revision"],
                    "prop_upward_revisions": row["prop_upward_revisions"],
                    "feat_claim_type": CLAIM_TYPE_NAMES.index(claim["claim_type"]),
                    "feat_litigation": float(claim["litigation"]),
                    "feat_fault": claim["fault"],
                })
        return panel

    def _add_split(self, df: pl.DataFrame) -> pl.DataFrame:
        """Assign train/test splits by claim_id (all rows for a claim go to same split)."""
        unique_claims = df["claim_id"].unique().to_list()
        rng = np.random.default_rng(self.random_state)
        rng.shuffle(unique_claims)
        n_train = int(len(unique_claims) * self.train_frac)
        train_ids = set(unique_claims[:n_train])

        split_series = pl.Series(
            "split",
            ["train" if cid in train_ids else "test" for cid in df["claim_id"].to_list()]
        )
        return df.with_columns(split_series)

    @staticmethod
    def _quarter_to_date(base_date: date, quarter_offset: int) -> date:
        """Convert a quarter offset to a date (first day of that quarter)."""
        total_months = base_date.month - 1 + quarter_offset * 3
        year = base_date.year + total_months // 12
        month = (total_months % 12) + 1
        return date(year, month, 1)
