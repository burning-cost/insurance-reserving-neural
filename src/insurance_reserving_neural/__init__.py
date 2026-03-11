"""
insurance-reserving-neural
==========================

Individual neural claims reserving for UK insurance pricing and reserving teams.

Per-claim models that predict outstanding liability, replacing aggregate
development triangles. Implements FNN+ (feedforward with case estimate features)
as the default architecture, following Avanzi et al. (2025) who showed case
estimate summaries outperform sequence memory at a fraction of the training cost.

Quick start
-----------
>>> from insurance_reserving_neural.synthetic import SyntheticClaimsGenerator
>>> from insurance_reserving_neural.models import FNNReserver
>>> from insurance_reserving_neural.metrics import mean_absolute_log_error, ocl_error
>>>
>>> gen = SyntheticClaimsGenerator(n_claims=5000, random_state=42)
>>> claims_df = gen.generate()
>>> train_df = claims_df.filter(pl.col("split") == "train")
>>> test_df  = claims_df.filter(pl.col("split") == "test")
>>>
>>> model = FNNReserver(use_case_estimates=True)
>>> model.fit(train_df)
>>> preds = model.predict(test_df)
>>> print(ocl_error(preds))

References
----------
Avanzi, Lambrianidis, Taylor, Wong (2025). arXiv:2601.05274
Richman, Wüthrich (2026). arXiv:2602.15385
Schneider, Schwab (2025). SSRN:4769020
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("insurance-reserving-neural")
except PackageNotFoundError:
    __version__ = "0.0.0"

from insurance_reserving_neural.data import ClaimSchema, prepare_features
from insurance_reserving_neural.synthetic import SyntheticClaimsGenerator
from insurance_reserving_neural.models import FNNReserver, LSTMReserver
from insurance_reserving_neural.loss import log_space_mse, duan_smearing_correction
from insurance_reserving_neural.metrics import (
    mean_absolute_log_error,
    ocl_error,
    reserve_range,
    chain_ladder_reserve,
)
from insurance_reserving_neural.bootstrap import BootstrapReserver

__all__ = [
    "__version__",
    "ClaimSchema",
    "prepare_features",
    "SyntheticClaimsGenerator",
    "FNNReserver",
    "LSTMReserver",
    "log_space_mse",
    "duan_smearing_correction",
    "mean_absolute_log_error",
    "ocl_error",
    "reserve_range",
    "chain_ladder_reserve",
    "BootstrapReserver",
]
