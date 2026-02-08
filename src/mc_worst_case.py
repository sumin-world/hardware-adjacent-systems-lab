#!/usr/bin/env python3
"""
mc_worst_case.py — Vectorized Monte Carlo worst-case analysis.

Generates N perturbed S21 traces in a single NumPy-vectorized pass (no
Python loop over samples), extracts the worst-case sample, and reports
the full objective distribution with percentiles.

Outputs:
    mc_report.json      — Distribution statistics, worst-case params
    mc_objectives.csv   — Per-sample objective values
    mc_hist.png         — Objective histogram with percentile markers
    worst_overlay.png   — Baseline vs worst-case S21 overlay
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import (
    ObjectiveMode,
    band_mask,
    compute_scalar_objective,
    load_network,
    percentile_stats,
    perturb_network_single,
    perturb_network_vectorized,
    write_json,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})


def _compute_objectives_vectorized(
    s21_db_all: np.ndarray,
    f_ghz: np.ndarray,
    fmin: float | None,
    fmax: float | None,
    mode: ObjectiveMode,
) -> np.ndarray:
    """Compute scalar objective for each of N traces — fully vectorized.

    Parameters
    ----------
    s21_db_all : (N, F) array
    f_ghz : (F,) array
    """
    mask = band_mask(f_ghz, fmin, fmax)
    s21_band = s21_db_all[:, mask]  # (N, F_band)

    if mode == "s21_min":
        return s21_band.min(axis=1)  # (N,)
    if mode == "s21_mean":
        return s21_band.mean(axis=1)
    raise ValueError(f"Unknown mode: {mode!r}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Vectorized Monte Carlo worst-case S-parameter analysis.",
    )
    ap.add_argument("--s2p", required=True, help="Path to .s2p file")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--n", type=int, default=2000, help="Number of MC samples")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--fmin", type=float, default=None, help="Band lower bound (GHz)")
    ap.add_argument("--fmax", type=float, default=None, help="Band upper bound (GHz)")
    ap.add_argument("--objective", default="s21_min",
                     choices=["s21_min", "s21_mean"], help="Scalar objective")
    ap.add_argument("--loss-mu-db", type=float, default=0.0)
    ap.add_argument("--loss-sigma-db", type=float, default=0.5)
    ap.add_argument("--slope-sigma-db", type=float, default=0.2)
    ap.add_argument("--refl-mu-db", type=float, default=0.0)
    ap.add_argument("--refl-sigma-db", type=float, default=0.5)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = load_network(args.s2p)
    f_ghz = base.f / 1e9
    rng = np.random.default_rng(args.seed)

    # ── Vectorized MC ─────────────────────────────────────────────────────
    t0 = time.perf_counter()

    s21_db_all, params_all = perturb_network_vectorized(
        base, args.n, rng,
        loss_mu_db=args.loss_mu_db,
        loss_sigma_db=args.loss_sigma_db,
        slope_sigma_db=args.slope_sigma_db,
        refl_mu_db=args.refl_mu_db,
        refl_sigma_db=args.refl_sigma_db,
    )

    objs = _compute_objectives_vectorized(
        s21_db_all, f_ghz, args.fmin, args.fmax, args.objective,
    )

    elapsed = time.perf_counter() - t0
    logger.info("MC done: %d samples in %.3f s (%.0f samples/s)",
                args.n, elapsed, args.n / elapsed)

    # ── Worst-case extraction ─────────────────────────────────────────────
    worst_idx = int(np.argmin(objs))
    worst_obj = float(objs[worst_idx])
    worst_params = {
        "a_loss_db": float(params_all[worst_idx, 0]),
        "b_slope_db": float(params_all[worst_idx, 1]),
        "refl_db": float(params_all[worst_idx, 2]),
    }

    base_obj = compute_scalar_objective(
        f_ghz, base.s_db[:, 1, 0], args.fmin, args.fmax, args.objective,
    )

    # ── Report ────────────────────────────────────────────────────────────
    stats = percentile_stats(objs)
    stats["worst"] = worst_obj

    report = {
        "input": str(args.s2p),
        "n": args.n,
        "seed": args.seed,
        "band_ghz": {"fmin": args.fmin, "fmax": args.fmax},
        "objective": args.objective,
        "baseline_obj": base_obj,
        "obj_stats": stats,
        "worst": {"idx": worst_idx, "obj": worst_obj, "params": worst_params},
        "perturb_params": {
            "loss_mu_db": args.loss_mu_db,
            "loss_sigma_db": args.loss_sigma_db,
            "slope_sigma_db": args.slope_sigma_db,
            "refl_mu_db": args.refl_mu_db,
            "refl_sigma_db": args.refl_sigma_db,
        },
        "runtime_s": round(elapsed, 4),
    }
    write_json(out_dir / "mc_report.json", report)

    # ── CSV ───────────────────────────────────────────────────────────────
    csv_path = out_dir / "mc_objectives.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample", "objective", "a_loss_db", "b_slope_db", "refl_db"])
        for i in range(args.n):
            w.writerow([i, float(objs[i]),
                        float(params_all[i, 0]),
                        float(params_all[i, 1]),
                        float(params_all[i, 2])])
    logger.info("Wrote %s", csv_path)

    # ── Histogram ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(objs, bins=min(80, args.n // 10), color="#4c72b0", edgecolor="white",
            linewidth=0.5, alpha=0.85)

    # Percentile markers
    for pct, color, ls in [(1, "#d62728", "-"), (5, "#ff7f0e", "--"), (50, "#2ca02c", ":")]:
        val = np.percentile(objs, pct)
        ax.axvline(val, color=color, ls=ls, lw=1.5, label=f"p{pct:02d} = {val:.2f} dB")

    ax.axvline(worst_obj, color="#d62728", ls="-", lw=2,
               label=f"worst = {worst_obj:.2f} dB")
    ax.set_title(f"Monte Carlo Objective Distribution  (N={args.n})")
    ax.set_xlabel(f"{args.objective} (dB)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "mc_hist.png", dpi=150)
    plt.close(fig)

    # ── Worst overlay ─────────────────────────────────────────────────────
    worst_ntwk = perturb_network_single(
        base,
        a_loss_db=worst_params["a_loss_db"],
        b_slope_db=worst_params["b_slope_db"],
        refl_db=worst_params["refl_db"],
    )
    mask = band_mask(f_ghz, args.fmin, args.fmax)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(f_ghz[mask], base.s_db[:, 1, 0][mask],
            lw=2, color="#1f77b4", label="Baseline S21")
    ax.plot(f_ghz[mask], worst_ntwk.s_db[:, 1, 0][mask],
            lw=2, color="#d62728", label=f"Worst-case S21 (sample #{worst_idx})")
    ax.set_title("Baseline vs Worst-Case (S21)")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("S21 (dB)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "worst_overlay.png", dpi=150)
    plt.close(fig)

    logger.info("Done → %s/mc_report.json, mc_objectives.csv, mc_hist.png, worst_overlay.png", out_dir)


if __name__ == "__main__":
    main()
