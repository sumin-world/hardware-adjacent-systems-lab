#!/usr/bin/env python3
"""
doe_runner.py — 2^k full-factorial DOE over perturbation σ parameters.

Runs a compact Monte Carlo at each DOE point and collects distribution
statistics. Computes main effects to rank which variation source has the
largest impact on worst-case performance — critical for allocating
manufacturing tolerance budgets.

Outputs:
    doe_report.json      — DOE configuration and main effects
    doe_results.csv      — Per-point MC summary statistics
    doe_sensitivity.png  — Main-effect Pareto chart
"""

from __future__ import annotations

import argparse
import csv
import itertools
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import (
    band_mask,
    load_network,
    percentile_stats,
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


def _run_mc_point(
    base,
    f_ghz: np.ndarray,
    mask: np.ndarray,
    seed: int,
    n: int,
    objective: str,
    loss_sigma_db: float,
    slope_sigma_db: float,
    refl_sigma_db: float,
) -> dict:
    """Run a single MC point (vectorized) and return summary stats."""
    rng = np.random.default_rng(seed)
    s21_db_all, params_all = perturb_network_vectorized(
        base, n, rng,
        loss_sigma_db=loss_sigma_db,
        slope_sigma_db=slope_sigma_db,
        refl_sigma_db=refl_sigma_db,
    )
    s21_band = s21_db_all[:, mask]  # (N, F_band)

    if objective == "s21_min":
        objs = s21_band.min(axis=1)
    else:
        objs = s21_band.mean(axis=1)

    worst_idx = int(np.argmin(objs))
    return {
        **percentile_stats(objs),
        "worst": float(objs[worst_idx]),
        "worst_params": {
            "a_loss_db": float(params_all[worst_idx, 0]),
            "b_slope_db": float(params_all[worst_idx, 1]),
            "refl_db": float(params_all[worst_idx, 2]),
        },
    }


def _compute_main_effects(
    rows: list[dict],
    factor_keys: list[str],
    response_key: str = "p01",
) -> dict[str, float]:
    """Compute 2-level factorial main effects.

    Main effect = mean(response at high) - mean(response at low).
    A more negative main effect means that factor worsens the objective
    more when increased, indicating higher sensitivity.
    """
    effects = {}
    for key in factor_keys:
        vals = sorted(set(r[key] for r in rows))
        if len(vals) != 2:
            continue
        lo, hi = vals
        resp_lo = np.mean([r[response_key] for r in rows if r[key] == lo])
        resp_hi = np.mean([r[response_key] for r in rows if r[key] == hi])
        effects[key] = float(resp_hi - resp_lo)
    return effects


def main() -> None:
    ap = argparse.ArgumentParser(
        description="2^k full-factorial DOE over perturbation sigma parameters.",
    )
    ap.add_argument("--s2p", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n", type=int, default=500,
                     help="MC samples per DOE point")
    ap.add_argument("--fmin", type=float, default=None)
    ap.add_argument("--fmax", type=float, default=None)
    ap.add_argument("--objective", default="s21_min",
                     choices=["s21_min", "s21_mean"])
    ap.add_argument("--loss-sigma-low", type=float, default=0.3)
    ap.add_argument("--loss-sigma-high", type=float, default=0.7)
    ap.add_argument("--slope-sigma-low", type=float, default=0.1)
    ap.add_argument("--slope-sigma-high", type=float, default=0.3)
    ap.add_argument("--refl-sigma-low", type=float, default=0.3)
    ap.add_argument("--refl-sigma-high", type=float, default=0.7)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = load_network(args.s2p)
    f_ghz = base.f / 1e9
    mask = band_mask(f_ghz, args.fmin, args.fmax)

    levels = {
        "loss_sigma_db": [args.loss_sigma_low, args.loss_sigma_high],
        "slope_sigma_db": [args.slope_sigma_low, args.slope_sigma_high],
        "refl_sigma_db": [args.refl_sigma_low, args.refl_sigma_high],
    }
    keys = list(levels.keys())
    combos = list(itertools.product(*[levels[k] for k in keys]))

    t0 = time.perf_counter()
    rows: list[dict] = []

    for i, (ls, ss, rs) in enumerate(combos):
        summary = _run_mc_point(
            base, f_ghz, mask,
            seed=args.seed + i,
            n=args.n,
            objective=args.objective,
            loss_sigma_db=ls,
            slope_sigma_db=ss,
            refl_sigma_db=rs,
        )
        row = {
            "run": i,
            "loss_sigma_db": ls,
            "slope_sigma_db": ss,
            "refl_sigma_db": rs,
            "p01": summary["p01"],
            "p05": summary["p05"],
            "p50": summary["p50"],
            "p95": summary["p95"],
            "worst": summary["worst"],
            "mean": summary["mean"],
            "var": summary["var"],
        }
        rows.append(row)
        logger.info("DOE point %d/%d  (σ_loss=%.2f, σ_slope=%.2f, σ_refl=%.2f)  p01=%.3f dB",
                     i + 1, len(combos), ls, ss, rs, summary["p01"])

    elapsed = time.perf_counter() - t0
    logger.info("DOE complete: %d points × %d samples in %.1f s", len(combos), args.n, elapsed)

    # ── CSV ───────────────────────────────────────────────────────────────
    csv_path = out_dir / "doe_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.info("Wrote %s", csv_path)

    # ── Main effects ──────────────────────────────────────────────────────
    effects = _compute_main_effects(rows, keys, response_key="p01")

    report = {
        "input": str(args.s2p),
        "objective": args.objective,
        "band_ghz": {"fmin": args.fmin, "fmax": args.fmax},
        "n_per_point": args.n,
        "levels": levels,
        "main_effects_p01": effects,
        "runtime_s": round(elapsed, 2),
        "notes": "2-level full factorial DOE; response = MC p01 of objective. "
                 "Main effect = mean(high) - mean(low); more negative = higher sensitivity.",
    }
    write_json(out_dir / "doe_report.json", report)

    # ── Sensitivity Pareto chart ──────────────────────────────────────────
    labels = list(effects.keys())
    values = [abs(effects[k]) for k in labels]
    sort_idx = np.argsort(values)[::-1]
    labels = [labels[i] for i in sort_idx]
    values = [values[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#d62728", "#ff7f0e", "#2ca02c"][:len(labels)]
    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.5)
    ax.set_xlabel("|Main Effect| on p01 (dB)")
    ax.set_title("DOE Sensitivity — Which σ Drives Worst-Case?")
    ax.invert_yaxis()

    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f} dB", va="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_dir / "doe_sensitivity.png", dpi=150)
    plt.close(fig)

    logger.info("Done → %s/doe_report.json, doe_results.csv, doe_sensitivity.png", out_dir)


if __name__ == "__main__":
    main()
