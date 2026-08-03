"""Data sources: tokenized .bin shards, plus two synthetic tasks for tests.

``induction`` is the one that matters for smoke testing: each sequence is a
random half repeated verbatim, so the second half is perfectly predictable by
an induction head and the loss has a known floor (~ln(V)/2).  A run that does
not beat that floor has a broken training loop, not a hard dataset.
"""

from __future__ import annotations

import os

import numpy as np


class BinData:
    """Memory-mapped uint16 token stream, nanoGPT layout."""

    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"token file {path!r} not found - run scripts/prepare_data.py first"
            )
        self.path = path
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")

    def __len__(self) -> int:
        return len(self.tokens)

    def batch(self, rng: np.random.Generator, batch_size: int, seq_len: int):
        hi = len(self.tokens) - seq_len - 1
        if hi <= 0:
            raise ValueError(f"{self.path} holds {len(self.tokens)} tokens, need > {seq_len + 1}")
        idx = rng.integers(0, hi, size=batch_size)
        x = np.stack([self.tokens[i : i + seq_len] for i in idx]).astype(np.int32)
        y = np.stack([self.tokens[i + 1 : i + 1 + seq_len] for i in idx]).astype(np.int32)
        return x, y


class InductionData:
    """A random pattern of length ``period``, tiled across the sequence.

    Everything after the first period is copyable from ``period`` positions
    back, so an attention head that learns one fixed offset solves the task.
    Keep ``period`` well below ``seq_len``: with a long period most positions
    carry no signal and finding the offset becomes a needle-in-a-haystack
    search that takes far longer than a smoke test should.
    """

    def __init__(self, vocab_size: int, seq_len: int, period: int = 8, n_special: int = 1):
        if period >= seq_len:
            raise ValueError(f"period {period} must be < seq_len {seq_len}")
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.period = period
        self.n_special = n_special

    @property
    def loss_floor(self) -> float:
        """Best achievable mean CE: only the first period is irreducible."""
        unpredictable = self.period - 1
        return float(np.log(self.vocab_size - self.n_special) * unpredictable / self.seq_len)

    def batch(self, rng: np.random.Generator, batch_size: int, seq_len: int):
        n = seq_len + 1
        reps = int(np.ceil(n / self.period))
        r = rng.integers(self.n_special, self.vocab_size, size=(batch_size, self.period))
        seq = np.tile(r, (1, reps))[:, :n].astype(np.int32)
        return seq[:, :-1], seq[:, 1:]


class SyntheticData:
    """Uniform random tokens: no learnable structure, throughput probe only."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    @property
    def loss_floor(self) -> float:
        return float(np.log(self.vocab_size))

    def batch(self, rng: np.random.Generator, batch_size: int, seq_len: int):
        seq = rng.integers(0, self.vocab_size, size=(batch_size, seq_len + 1)).astype(np.int32)
        return seq[:, :-1], seq[:, 1:]


def build_data(tc, mc):
    """Return ``(train_source, val_source)`` for a config pair."""
    if tc.dataset == "bin":
        train = tc.train_bin or os.path.join(tc.data_dir, "train.bin")
        val = tc.val_bin or os.path.join(tc.data_dir, "val.bin")
        return BinData(train), BinData(val)
    if tc.dataset == "induction":
        d = InductionData(mc.vocab_size, mc.seq_len, tc.induction_period)
        return d, d
    if tc.dataset == "synthetic":
        d = SyntheticData(mc.vocab_size)
        return d, d
    raise ValueError(f"unknown dataset {tc.dataset!r}")


def loss_floor(source) -> float | None:
    return getattr(source, "loss_floor", None)
