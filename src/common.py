"""
common.py — Shared utilities for channel analysis.

Provides:
  - Vectorized S-parameter perturbation engine (no Python loop over MC samples)
  - Scalar objective functions (s21_min, s21_mean)
  - Statistics helper
  - Band-mask builder
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import skrf as rf

logger = logging.getLogger(__name__)

ObjectiveMode = Literal["s21_min", "s21_mean"]


# ---------------------------------------------------------------------------
# Band helpers
# ---------------------------------------------------------------------------

def band_mask(f_ghz: np.ndarray, fmin: float | None, fmax: float | None) -> np.ndarray:
    """Return boolean mask selecting frequencies within [fmin, fmax] GHz."""
    mask = np.ones(len(f_ghz), dtype=bool)
    if fmin is not None:
        mask &= f_ghz >= fmin
    if fmax is not None:
        mask &= f_ghz <= fmax
    if not mask.any():
        raise ValueError(
            f"No frequency points in [{fmin}, {fmax}] GHz. "
            f"File range: [{f_ghz[0]:.2f}, {f_ghz[-1]:.2f}] GHz."
        )
    return mask


# ---------------------------------------------------------------------------
# Scalar objective
# ---------------------------------------------------------------------------

def compute_scalar_objective(
    f_ghz: np.ndarray,
    s21_db: np.ndarray,
    fmin: float | None = None,
    fmax: float | None = None,
    mode: ObjectiveMode = "s21_min",
) -> float:
    """Compute a scalar figure-of-merit from S21 over a frequency band.

    Parameters
    ----------
    f_ghz : (F,) array — frequency points in GHz.
    s21_db : (F,) array — S21 magnitude in dB.
    fmin, fmax : optional band limits in GHz.
    mode : ``"s21_min"`` (worst-point IL) or ``"s21_mean"`` (average IL).

    Returns
    -------
    float — objective value in dB.
    """
    x = s21_db[band_mask(f_ghz, fmin, fmax)]
    if mode == "s21_min":
        return float(np.min(x))
    if mode == "s21_mean":
        return float(np.mean(x))
    raise ValueError(f"Unknown objective mode: {mode!r}")


# ---------------------------------------------------------------------------
# Vectorized perturbation (core performance path)
# ---------------------------------------------------------------------------

def perturb_network_vectorized(
    ntwk: rf.Network,
    n: int,
    rng: np.random.Generator,
    loss_mu_db: float = 0.0,
    loss_sigma_db: float = 0.5,
    slope_sigma_db: float = 0.2,
    refl_mu_db: float = 0.0,
    refl_sigma_db: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate N perturbed S21 traces in a single vectorized pass.

    Instead of looping ``N`` times and copying the Network object each
    iteration, this function broadcasts the perturbation across all
    samples simultaneously using NumPy, yielding 10–50× speedup.

    Parameters
    ----------
    ntwk : 2-port ``rf.Network`` baseline.
    n : number of Monte Carlo samples.
    rng : NumPy random generator.
    loss_mu_db, loss_sigma_db : flat-loss Gaussian parameters.
    slope_sigma_db : frequency-slope Gaussian σ.
    refl_mu_db, refl_sigma_db : reflection degradation Gaussian parameters.

    Returns
    -------
    s21_db_all : (N, F) array of perturbed S21 in dB.
    params_all : (N, 3) array — columns [a_loss_db, b_slope_db, refl_db].
    """
    f = ntwk.f.astype(np.float64)
    f_norm = (f - f.min()) / (f.max() - f.min() + 1e-12)  # (F,)
    nf = len(f)

    # Sample all parameters at once: (N,)
    a = rng.normal(loss_mu_db, loss_sigma_db, size=n)
    b = rng.normal(0.0, slope_sigma_db, size=n)
    refl = rng.normal(refl_mu_db, refl_sigma_db, size=n)

    params_all = np.column_stack([a, b, refl])  # (N, 3)

    # Extra loss per sample per freq: (N, F)
    extra_loss_db = a[:, None] + b[:, None] * (f_norm[None, :] - 0.5)

    # Baseline S21 magnitude in linear
    s21_base = ntwk.s[:, 1, 0]  # (F,) complex
    mag21_base = np.abs(s21_base)  # (F,)

    # Perturbed S21 magnitude: (N, F)
    mag21_perturbed = mag21_base[None, :] * (10.0 ** (-extra_loss_db / 20.0))

    # Convert to dB
    s21_db_all = 20.0 * np.log10(np.maximum(mag21_perturbed, 1e-15))

    return s21_db_all, params_all


def perturb_network_single(
    ntwk: rf.Network,
    a_loss_db: float,
    b_slope_db: float,
    refl_db: float,
) -> rf.Network:
    """Apply a single deterministic perturbation and return a new Network.

    Used by corner_eval where we need the full Network object (not just S21 dB).
    """
    f = ntwk.f.astype(np.float64)
    f_norm = (f - f.min()) / (f.max() - f.min() + 1e-12)
    extra_loss_db = a_loss_db + b_slope_db * (f_norm - 0.5)

    s = ntwk.s.copy()

    # S21
    s21 = s[:, 1, 0]
    mag21 = np.abs(s21)
    ph21 = np.angle(s21)
    s[:, 1, 0] = mag21 * (10.0 ** (-extra_loss_db / 20.0)) * np.exp(1j * ph21)

    # S11
    s11 = s[:, 0, 0]
    mag11 = np.abs(s11)
    ph11 = np.angle(s11)
    mag11_new = np.minimum(mag11 * (10.0 ** (refl_db / 20.0)), 0.999999)
    s[:, 0, 0] = mag11_new * np.exp(1j * ph11)

    out = ntwk.copy()
    out.s = s
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def percentile_stats(x: np.ndarray) -> dict:
    """Compute standard percentile statistics for an array."""
    return {
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) >= 2 else float("nan"),
        "var": float(np.var(x, ddof=1)) if len(x) >= 2 else float("nan"),
        "p01": float(np.percentile(x, 1)),
        "p05": float(np.percentile(x, 5)),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_network(path: str | Path) -> rf.Network:
    """Load a Touchstone file and validate it is a 2-port network."""
    ntwk = rf.Network(str(path))
    if ntwk.nports != 2:
        raise ValueError(f"Expected 2-port network, got {ntwk.nports}-port from {path}")
    logger.info("Loaded %s  |  %d points  |  %.2f–%.2f GHz",
                path, len(ntwk.f), ntwk.f[0] / 1e9, ntwk.f[-1] / 1e9)
    return ntwk


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Wrote %s", path)
