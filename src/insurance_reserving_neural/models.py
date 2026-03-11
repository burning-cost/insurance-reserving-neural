"""
models.py — FNNReserver and LSTMReserver for individual claims reserving.

Two model classes, both with the same sklearn-style fit/predict/reserve API:

FNNReserver (default)
    Feedforward neural network with summarised claim features. When
    use_case_estimates=True this is the "FNN+" of Avanzi et al. (2025),
    which adds case estimate history summaries to the feature vector. FNN+
    nearly matches LSTM+ but trains in ~2 minutes on CPU vs ~15 minutes.
    This is the right default for most UK reserving teams.

LSTMReserver (optional, requires torch)
    LSTM over the claim payment/case estimate sequence. Handles variable-length
    claim histories via pack_padded_sequence. Takes ~4x longer to train but
    may help on portfolios with very long development tails (e.g., large BI claims
    with 10+ year development). Optional because it requires torch and is slower.

Both models predict log(outstanding) and apply Duan (1983) smearing bias
correction on back-transformation. Both produce per-claim reserve predictions
and aggregate portfolio RBNS reserves.

Design decisions
----------------
- We use PyTorch throughout rather than sklearn MLPRegressor because:
  (a) gradient clipping is essential for heavy-tailed insurance data
  (b) LSTM requires custom sequence handling that sklearn cannot do
  (c) custom loss functions and early stopping logic
- All input features are standardised (zero mean, unit variance) internally
- Batch normalisation on FNN layers; layer normalisation on LSTM layers
- AdamW optimiser with weight decay prevents overfitting on small portfolios
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple
import numpy as np
import polars as pl

from insurance_reserving_neural.data import (
    prepare_features,
    get_feature_columns,
    to_numpy_features,
    to_numpy_targets,
)
from insurance_reserving_neural.loss import (
    duan_smearing_correction,
    apply_smearing,
)


def _require_torch():
    """Lazy import of torch with a clear error message."""
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for neural network models. "
            "Install it with: pip install insurance-reserving-neural[torch]\n"
            "Or: pip install torch>=2.0"
        )


class _FNNNet:
    """
    Pure PyTorch module wrapper — kept separate from the sklearn-style
    FNNReserver so it can be imported only when torch is available.
    """

    def __new__(cls, *args, **kwargs):
        torch = _require_torch()
        nn = torch.nn

        class _Net(nn.Module):
            def __init__(self, n_features: int, hidden_sizes: List[int], dropout: float):
                super().__init__()
                layers = []
                in_dim = n_features
                for h in hidden_sizes:
                    layers.extend([
                        nn.Linear(in_dim, h),
                        nn.BatchNorm1d(h),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ])
                    in_dim = h
                layers.append(nn.Linear(in_dim, 1))
                self.net = nn.Sequential(*layers)

            def forward(self, x):
                return self.net(x).squeeze(-1)

        return _Net(*args, **kwargs)


class FNNReserver:
    """
    Feedforward neural network for individual RBNS claims reserving.

    When use_case_estimates=True (default), this is the FNN+ architecture
    from Avanzi, Lambrianidis, Taylor, Wong (2025). The case estimate summary
    features (n_revisions, mean CE, std CE, trend, largest revision,
    proportion upward) add 15-25% MALE improvement over plain FNN.

    Parameters
    ----------
    use_case_estimates : bool
        Include case estimate history features (FNN+). Default True.
    hidden_sizes : tuple of int
        Hidden layer sizes. Default (64, 32, 16) — slightly wider than
        Avanzi et al.'s (64, 32) for richer claim type heterogeneity.
    dropout : float
        Dropout probability on hidden layers. Default 0.1.
    learning_rate : float
        AdamW learning rate. Default 0.001.
    weight_decay : float
        AdamW weight decay (L2 regularisation). Default 1e-4.
    max_epochs : int
        Maximum training epochs. Default 200. Early stopping usually
        triggers well before this.
    patience : int
        Early stopping patience (epochs without improvement). Default 10.
    batch_size : int
        Mini-batch size. Default 512.
    grad_clip : float
        Gradient clipping max norm. Essential for heavy-tailed insurance data.
        Default 1.0 (from Avanzi et al. 2025).
    val_frac : float
        Fraction of training data held out for early stopping validation.
        Default 0.15.
    random_state : int | None
        Random seed.
    verbose : bool
        Print training progress. Default False.
    """

    def __init__(
        self,
        use_case_estimates: bool = True,
        hidden_sizes: Tuple[int, ...] = (64, 32, 16),
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 200,
        patience: int = 10,
        batch_size: int = 512,
        grad_clip: float = 1.0,
        val_frac: float = 0.15,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        self.use_case_estimates = use_case_estimates
        self.hidden_sizes = list(hidden_sizes)
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.val_frac = val_frac
        self.random_state = random_state
        self.verbose = verbose

        self._net = None
        self._feature_cols: List[str] = []
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None
        self._smearing_factor: float = 1.0
        self._is_fitted: bool = False
        self.training_history_: List[dict] = []

    def fit(self, df: pl.DataFrame) -> "FNNReserver":
        """
        Train the FNN on settled claims.

        Only settled claims (is_open == False) with known ultimate are used
        for training. Open claims are predictions targets only.

        Parameters
        ----------
        df : pl.DataFrame
            Training data in panel format. Must contain ultimate column for
            settled claims.

        Returns
        -------
        self
        """
        torch = _require_torch()

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # Prepare features
        df = prepare_features(df, use_case_estimates=self.use_case_estimates)

        # Training data: all rows where the eventual ultimate is known and
        # outstanding > 0. This includes open claims at intermediate quarters of
        # eventually-settled claims, not just the final settled row.
        # The key invariant: ultimate > cumulative_paid (positive outstanding to predict).
        settled = df.filter(
            pl.col("ultimate").is_not_null()
            & (pl.col("ultimate") > pl.col("cumulative_paid"))
        )

        if len(settled) < 100:
            raise ValueError(
                f"Only {len(settled)} training observations with positive outstanding found. "
                "FNNReserver needs at least 100 observations. "
                "Ensure your training data has ultimate values and claims with positive outstanding."
            )

        self._feature_cols = get_feature_columns(settled, self.use_case_estimates)

        if len(self._feature_cols) == 0:
            raise ValueError("No feature columns found. Check that prepare_features() ran successfully.")

        X = to_numpy_features(settled, self._feature_cols)
        y = to_numpy_targets(settled)

        # Standardise features
        self._feature_mean = X.mean(axis=0)
        self._feature_std = X.std(axis=0) + 1e-8
        X_std = (X - self._feature_mean) / self._feature_std

        # Train/val split
        n = len(X_std)
        n_val = max(1, int(n * self.val_frac))
        idx = np.random.permutation(n) if self.random_state is None else np.random.RandomState(self.random_state).permutation(n)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

        X_train, y_train = X_std[train_idx], y[train_idx]
        X_val, y_val = X_std[val_idx], y[val_idx]

        # Build network
        n_features = X_train.shape[1]
        self._net = _FNNNet(n_features, self.hidden_sizes, self.dropout)

        optimizer = torch.optim.AdamW(
            self._net.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = torch.nn.MSELoss()

        X_train_t = torch.tensor(X_train)
        y_train_t = torch.tensor(y_train)
        X_val_t = torch.tensor(X_val)
        y_val_t = torch.tensor(y_val)

        best_val_loss = float("inf")
        best_state = None
        patience_count = 0

        for epoch in range(self.max_epochs):
            self._net.train()
            # Shuffle
            perm = torch.randperm(len(X_train_t))
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(X_train_t), self.batch_size):
                batch_idx = perm[start:start + self.batch_size]
                xb = X_train_t[batch_idx]
                yb = y_train_t[batch_idx]

                optimizer.zero_grad()
                pred = self._net(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), self.grad_clip)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # Validation
            self._net.eval()
            with torch.no_grad():
                val_pred = self._net(X_val_t)
                val_loss = loss_fn(val_pred, y_val_t).item()

            train_loss = epoch_loss / max(1, n_batches)
            self.training_history_.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            if self.verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)

        # Compute Duan smearing factor on full training set
        self._net.eval()
        with torch.no_grad():
            X_all_t = torch.tensor(X_std)
            y_pred_log = self._net(X_all_t).numpy()

        self._smearing_factor = duan_smearing_correction(y, y_pred_log)

        self._is_fitted = True
        return self

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Predict outstanding liability for each claim row.

        For open claims this is the predicted RBNS outstanding.
        For settled claims it can be used for back-testing.

        Parameters
        ----------
        df : pl.DataFrame
            Claims in panel format (any mix of open and settled).

        Returns
        -------
        pl.DataFrame
            Original DataFrame with columns added:
            - predicted_outstanding : float — predicted outstanding liability
            - predicted_log_outstanding : float — raw log-scale prediction
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")

        torch = _require_torch()

        df = prepare_features(df, use_case_estimates=self.use_case_estimates)
        X = to_numpy_features(df, self._feature_cols)
        X_std = (X - self._feature_mean) / self._feature_std

        self._net.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_std)
            y_pred_log = self._net(X_t).numpy()

        predicted_outstanding = apply_smearing(y_pred_log, self._smearing_factor)
        predicted_outstanding = np.clip(predicted_outstanding, 0.0, None)

        return df.with_columns([
            pl.Series("predicted_outstanding", predicted_outstanding.astype(np.float64)),
            pl.Series("predicted_log_outstanding", y_pred_log.astype(np.float64)),
        ])

    def reserve(self, df: pl.DataFrame) -> float:
        """
        Compute total RBNS reserve: sum of predicted outstanding for open claims.

        Parameters
        ----------
        df : pl.DataFrame
            Claims panel containing open claims at the valuation date.

        Returns
        -------
        float
            Portfolio RBNS reserve estimate.
        """
        preds = self.predict(df)
        open_claims = preds.filter(pl.col("is_open") == True)
        return float(open_claims["predicted_outstanding"].sum())


class _LSTMNet:
    """Lazy-import PyTorch LSTM module."""

    def __new__(cls, *args, **kwargs):
        torch = _require_torch()
        nn = torch.nn

        class _Net(nn.Module):
            def __init__(
                self,
                input_size: int,
                static_size: int,
                hidden_size: int = 32,
                num_layers: int = 2,
                dropout: float = 0.1,
            ):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                self.layer_norm = nn.LayerNorm(hidden_size)
                # Merge LSTM output with static features
                merge_dim = hidden_size + static_size
                self.head = nn.Sequential(
                    nn.Linear(merge_dim, 32),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(32, 1),
                )

            def forward(self, seq, lengths, static):
                """
                seq    : (batch, max_len, input_size) — padded sequences
                lengths: (batch,) — actual sequence lengths (LongTensor)
                static : (batch, static_size) — static claim features
                """
                import torch
                packed = torch.nn.utils.rnn.pack_padded_sequence(
                    seq, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                out, (h_n, _) = self.lstm(packed)
                # Use last hidden state of last layer
                h_last = h_n[-1]  # (batch, hidden_size)
                h_last = self.layer_norm(h_last)
                merged = torch.cat([h_last, static], dim=1)
                return self.head(merged).squeeze(-1)

        return _Net(*args, **kwargs)


class LSTMReserver:
    """
    LSTM-based individual claims reserver.

    Processes the full sequence of (payment, case_estimate) pairs for each
    claim, using pack_padded_sequence to handle variable development lengths.
    Static claim features (claim type, litigation, fault, accident quarter)
    are merged with the LSTM final hidden state before the prediction head.

    Use this when you have portfolios with long development tails where the
    sequence of case estimate revisions carries predictive information beyond
    the summary statistics used by FNNReserver. For most UK portfolios,
    FNNReserver(use_case_estimates=True) matches this model's performance
    at 4-8x lower training cost.

    Parameters
    ----------
    hidden_size : int
        LSTM hidden state size. Default 32.
    num_layers : int
        Number of LSTM layers. Default 2.
    dropout : float
        Dropout on LSTM and head layers. Default 0.1.
    learning_rate : float
        AdamW learning rate. Default 0.001.
    weight_decay : float
        AdamW L2 regularisation. Default 1e-4.
    max_epochs : int
        Maximum training epochs. Default 100.
    patience : int
        Early stopping patience. Default 10.
    batch_size : int
        Claims per mini-batch. Default 128. Note: each claim becomes a
        variable-length sequence, so effective sequence tokens per batch
        is batch_size × avg_dev_quarters.
    grad_clip : float
        Gradient clipping max norm. Default 1.0.
    val_frac : float
        Validation fraction for early stopping. Default 0.15.
    random_state : int | None
        Random seed.
    verbose : bool
        Print training progress. Default False.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        patience: int = 10,
        batch_size: int = 128,
        grad_clip: float = 1.0,
        val_frac: float = 0.15,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.val_frac = val_frac
        self.random_state = random_state
        self.verbose = verbose

        self._net = None
        self._smearing_factor: float = 1.0
        self._seq_feature_mean: Optional[np.ndarray] = None
        self._seq_feature_std: Optional[np.ndarray] = None
        self._static_feature_mean: Optional[np.ndarray] = None
        self._static_feature_std: Optional[np.ndarray] = None
        self._is_fitted: bool = False
        self.training_history_: List[dict] = []

    # Sequence features: per-quarter values
    _SEQ_COLS = ["log_cumulative_paid", "case_estimate", "dev_quarter"]
    # Static features: claim-level (not time-varying)
    _STATIC_COLS = ["feat_claim_type", "feat_litigation", "feat_fault", "calendar_quarter"]

    def _build_sequences(
        self,
        df: pl.DataFrame,
        fit_mode: bool = False,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        """
        Build per-claim sequences from the panel DataFrame.

        For each claim, collects all development quarters into a sequence.
        Returns:
        - sequences: list of (T_i, n_seq_features) arrays
        - static_feats: list of (n_static_features,) arrays
        - targets: (n_claims,) log-outstanding at final observation
        """
        df = prepare_features(df, use_case_estimates=False)

        # Add log_cumulative_paid if not present
        seq_cols_avail = [c for c in self._SEQ_COLS if c in df.columns]
        static_cols_avail = [c for c in self._STATIC_COLS if c in df.columns]

        # Group by claim_id, sort by dev_quarter
        sequences = []
        static_feats = []
        targets = []

        claim_ids = df["claim_id"].unique().to_list()

        for cid in claim_ids:
            claim_df = df.filter(pl.col("claim_id") == cid).sort("dev_quarter")

            # In fit_mode: target = outstanding at last observed quarter.
            # The outer query guarantees ultimate > cumulative_paid at last row.
            if fit_mode:
                last_row = claim_df.tail(1)
                if last_row["ultimate"].is_null()[0]:
                    continue
                outstanding = float(last_row["ultimate"][0]) - float(last_row["cumulative_paid"][0])
                if outstanding <= 0:
                    continue
                targets.append(np.log(max(outstanding, 1e-6)))

            seq = claim_df.select(seq_cols_avail).to_numpy().astype(np.float32)
            sequences.append(seq)

            # Static features: take from last row
            static = claim_df.tail(1).select(static_cols_avail).to_numpy().astype(np.float32).squeeze()
            if static.ndim == 0:
                static = np.array([static], dtype=np.float32)
            static_feats.append(static)

        return sequences, static_feats, np.array(targets, dtype=np.float32)

    def fit(self, df: pl.DataFrame) -> "LSTMReserver":
        """
        Train the LSTM on settled claims.

        Parameters
        ----------
        df : pl.DataFrame
            Training panel. Settled claims (is_open=False) with known ultimate
            are used for training.

        Returns
        -------
        self
        """
        torch = _require_torch()

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # For LSTM training, use all claims that eventually settled.
        # Prepare features first so dev_quarter is available for sorting.
        df = prepare_features(df, use_case_estimates=False)

        settled_claim_ids = (
            df.sort("dev_quarter")
            .group_by("claim_id")
            .last()
            .filter(
                pl.col("ultimate").is_not_null()
                & (pl.col("ultimate") > pl.col("cumulative_paid"))
            )["claim_id"]
            .to_list()
        )
        settled_df = df.filter(pl.col("claim_id").is_in(settled_claim_ids))

        sequences, static_feats, y = self._build_sequences(settled_df, fit_mode=True)

        if len(sequences) < 50:
            raise ValueError(
                f"Only {len(sequences)} usable claim sequences found. "
                "LSTMReserver needs at least 50 claims with known ultimate > paid."
            )

        n_seq = sequences[0].shape[1]
        n_static = static_feats[0].shape[0]

        # Fit standardisation on flattened sequence values
        all_seq = np.vstack(sequences)
        self._seq_feature_mean = all_seq.mean(axis=0)
        self._seq_feature_std = all_seq.std(axis=0) + 1e-8

        all_static = np.vstack(static_feats)
        self._static_feature_mean = all_static.mean(axis=0)
        self._static_feature_std = all_static.std(axis=0) + 1e-8

        # Standardise
        seqs_std = [
            (s - self._seq_feature_mean) / self._seq_feature_std
            for s in sequences
        ]
        static_std = [
            (s - self._static_feature_mean) / self._static_feature_std
            for s in static_feats
        ]

        # Build network
        self._net = _LSTMNet(n_seq, n_static, self.hidden_size, self.num_layers, self.dropout)
        optimizer = torch.optim.AdamW(
            self._net.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = torch.nn.MSELoss()

        n = len(seqs_std)
        n_val = max(1, int(n * self.val_frac))
        idx = np.random.permutation(n)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

        def _make_batch(indices):
            batch_seqs = [seqs_std[i] for i in indices]
            batch_static = [static_std[i] for i in indices]
            batch_y = y[indices]
            lengths = torch.tensor([len(s) for s in batch_seqs], dtype=torch.long)
            max_len = max(len(s) for s in batch_seqs)
            padded = np.zeros((len(batch_seqs), max_len, n_seq), dtype=np.float32)
            for j, s in enumerate(batch_seqs):
                padded[j, :len(s)] = s
            seq_t = torch.tensor(padded)
            static_t = torch.tensor(np.array(batch_static, dtype=np.float32))
            y_t = torch.tensor(batch_y)
            return seq_t, lengths, static_t, y_t

        best_val_loss = float("inf")
        best_state = None
        patience_count = 0

        for epoch in range(self.max_epochs):
            self._net.train()
            perm = np.random.permutation(len(train_idx))
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(train_idx), self.batch_size):
                bidx = train_idx[perm[start:start + self.batch_size]]
                seq_t, lengths, static_t, y_t = _make_batch(bidx)

                optimizer.zero_grad()
                pred = self._net(seq_t, lengths, static_t)
                loss = loss_fn(pred, y_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), self.grad_clip)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            # Validation
            self._net.eval()
            with torch.no_grad():
                val_seq_t, val_lengths, val_static_t, val_y_t = _make_batch(val_idx)
                val_pred = self._net(val_seq_t, val_lengths, val_static_t)
                val_loss = loss_fn(val_pred, val_y_t).item()

            train_loss = epoch_loss / max(1, n_batches)
            self.training_history_.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            if self.verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)

        # Duan smearing factor
        self._net.eval()
        with torch.no_grad():
            all_seq_t, all_lengths, all_static_t, all_y_t = _make_batch(np.arange(len(seqs_std)))
            all_pred_log = self._net(all_seq_t, all_lengths, all_static_t).numpy()

        self._smearing_factor = duan_smearing_correction(y, all_pred_log)
        self._is_fitted = True
        return self

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Predict outstanding liability for each claim (using last observed dev quarter).

        Parameters
        ----------
        df : pl.DataFrame
            Claims in panel format.

        Returns
        -------
        pl.DataFrame
            One row per claim (at last valuation date) with columns:
            claim_id, predicted_outstanding, predicted_log_outstanding
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")

        torch = _require_torch()

        sequences, static_feats, _ = self._build_sequences(df, fit_mode=False)
        claim_ids = self._get_last_claim_ids(df)

        seqs_std = [
            (s - self._seq_feature_mean) / self._seq_feature_std
            for s in sequences
        ]
        static_std = [
            (s - self._static_feature_mean) / self._static_feature_std
            for s in static_feats
        ]

        n_seq = sequences[0].shape[1] if sequences else 0
        n_static = static_feats[0].shape[0] if static_feats else 0
        max_len = max(len(s) for s in seqs_std) if seqs_std else 1
        lengths = torch.tensor([len(s) for s in seqs_std], dtype=torch.long)
        padded = np.zeros((len(seqs_std), max_len, n_seq), dtype=np.float32)
        for j, s in enumerate(seqs_std):
            padded[j, :len(s)] = s
        seq_t = torch.tensor(padded)
        static_t = torch.tensor(np.array(static_std, dtype=np.float32))

        self._net.eval()
        with torch.no_grad():
            y_pred_log = self._net(seq_t, lengths, static_t).numpy()

        predicted = apply_smearing(y_pred_log, self._smearing_factor)
        predicted = np.clip(predicted, 0.0, None)

        return pl.DataFrame({
            "claim_id": claim_ids,
            "predicted_outstanding": predicted.astype(np.float64),
            "predicted_log_outstanding": y_pred_log.astype(np.float64),
        })

    def reserve(self, df: pl.DataFrame) -> float:
        """
        Portfolio RBNS reserve: sum of predicted outstanding for open claims.
        """
        # Get open claim IDs
        open_ids = set(df.filter(pl.col("is_open") == True)["claim_id"].to_list())
        preds = self.predict(df)
        open_preds = preds.filter(pl.col("claim_id").is_in(list(open_ids)))
        return float(open_preds["predicted_outstanding"].sum())

    def _get_last_claim_ids(self, df: pl.DataFrame) -> List[str]:
        """Return claim IDs in the same order as _build_sequences returns them."""
        return df["claim_id"].unique().to_list()
