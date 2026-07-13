"""Transparent helpers for discrete-time risk evaluation.

Frozen formal IPCW/bootstrap results are distributed as aggregate CSV files.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class HorizonMetrics:
    auroc: float
    auprc: float
    brier: float
    n: int
    events: int


def hazards_to_cumulative_risks(hazards: np.ndarray) -> np.ndarray:
    array = np.asarray(hazards, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError("hazards must have shape [n, 4].")
    if not np.isfinite(array).all():
        raise ValueError("hazards contain non-finite values.")
    if ((array < 0.0) | (array > 1.0)).any():
        raise ValueError("hazards must lie in [0, 1].")
    return 1.0 - np.cumprod(1.0 - array, axis=1)


def validate_monotonic_risks(risks: np.ndarray) -> None:
    array = np.asarray(risks, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError("risks must have shape [n, 4].")
    if (np.diff(array, axis=1) < -1e-12).any():
        raise ValueError("Cumulative risks are not monotonic.")


def evaluate_binary_horizon(outcomes: np.ndarray, probabilities: np.ndarray) -> HorizonMetrics:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    y = np.asarray(outcomes, dtype=np.int8)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.shape != p.shape:
        raise ValueError("outcomes and probabilities must have equal shape.")
    auroc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    return HorizonMetrics(
        auroc=auroc,
        auprc=float(average_precision_score(y, p)),
        brier=float(brier_score_loss(y, p)),
        n=int(len(y)),
        events=int(y.sum()),
    )


def patient_bootstrap_indices(patient_ids: np.ndarray, n_replicates: int = 1000, seed: int = 20260719) -> list[np.ndarray]:
    ids = np.asarray(patient_ids)
    unique = np.unique(ids)
    if len(unique) == 0:
        raise ValueError("No patient identifiers supplied.")
    lookup = {item: np.flatnonzero(ids == item) for item in unique}
    rng = np.random.default_rng(seed)
    output = []
    for _ in range(n_replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        output.append(np.concatenate([lookup[item] for item in sampled]))
    return output
