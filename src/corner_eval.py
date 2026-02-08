#!/usr/bin/env python3
"""
corner_eval.py — Deterministic corner sweep and MC gap analysis.

Evaluates all 2^3 = 8 sign combinations of (±k·σ_loss, ±k·σ_slope, ±k·σ_refl)
and checks whether the worst deterministic corner reaches (or exceeds) the
Monte Carlo statistical worst-case.

A positive ``miss_gap_db`` means corners failed to capture the MC tail —
indicating the need for more corners, higher k, or additional variation
parameters.

Outputs:
    corner_report.json       — Summary with MC comparison
    corner_table.csv         — All 8 corners with objective values
    worst_corner_overlay.png — Baseline vs worst-corner S21
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import (
    band_mask,
    compute_scalar_objective,
    load_network,
    perturb_network_single,
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Deterministic corner analysis with MC gap evaluation.",
    )
    ap.add_argument("--s2p", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fmin", type=float, default=None)
    ap.add_argument("--fmax", type=float, default=None)
    ap.add_argument("--objective", default="s21_min",
                     choices=["s21_min", "s21_mean"])
    ap.add_argument("--k", type=float, default=3.0,
                     help="Corner magnitude in units of sigma (default: 3σ)")
    ap.add_argument("--loss-sigma-db", type=float, default=0.5)
    ap.add_argument("--slope-sigma-db", type=float, default=0.2)
    ap.add_argument("--refl-sigma-db", type=float, default=0.5)
    ap.add_argument("--mc-report", default=None,
                     help="Path to mc_report.json for gap comparison")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = load_network(args.s2p)
    f_ghz = base.f / 1e9
    base_obj = compute_scalar_objective(
        f_ghz, base.s_db[:, 1, 0], args.fmin, args.fmax, args.objective,
    )

    # ── 2^3 corner sweep ──────────────────────────────────────────────────
    signs = list(itertools.product((-1, 1), repeat=3))
    sigmas = [args.loss_sigma_db, args.slope_sigma_db, args.refl_sigma_db]

    rows: list[dict] = []
    worst_obj = float("inf")
    worst_row: dict = {}
    worst_ntwk = None

    for idx, (sa, sb, sr) in enumerate(signs):
        a = sa * args.k * sigmas[0]
        b = sb * args.k * sigmas[1]
        r = sr * args.k * sigmas[2]

        nt = perturb_network_single(base, a, b, r)
        obj = compute_scalar_objective(
            f_ghz, nt.s_db[:, 1, 0], args.fmin, args.fmax, args.objective,
        )

        row = {
            "corner": idx,
            "a_loss_db": round(a, 4),
            "b_slope_db": round(b, 4),
            "refl_db": round(r, 4),
            "objective_db": round(obj, 6),
        }
        rows.append(row)

        if obj < worst_obj:
            worst_obj = obj
            worst_row = row
            worst_ntwk = nt

    # ── CSV ───────────────────────────────────────────────────────────────
    csv_path = out_dir / "corner_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.info("Wrote %s", csv_path)

    # ── Report ────────────────────────────────────────────────────────────
    report: dict = {
        "input": str(args.s2p),
        "band_ghz": {"fmin": args.fmin, "fmax": args.fmax},
        "objective": args.objective,
        "baseline_obj": base_obj,
        "k": args.k,
        "sigmas": {
            "loss_sigma_db": args.loss_sigma_db,
            "slope_sigma_db": args.slope_sigma_db,
            "refl_sigma_db": args.refl_sigma_db,
        },
        "n_corners": len(rows),
        "corner_worst": worst_row,
    }

    if args.mc_report:
        mc = json.loads(Path(args.mc_report).read_text(encoding="utf-8"))
        mc_worst = float(mc["obj_stats"]["worst"])
        gap = float(worst_obj - mc_worst)
        report["mc_worst"] = mc_worst
        report["miss_gap_db"] = round(gap, 6)
        logger.info("Corner worst: %.4f dB | MC worst: %.4f dB | Gap: %.4f dB",
                     worst_obj, mc_worst, gap)

    write_json(out_dir / "corner_report.json", report)

    # ── Overlay plot ──────────────────────────────────────────────────────
    mask = band_mask(f_ghz, args.fmin, args.fmax)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(f_ghz[mask], base.s_db[:, 1, 0][mask],
            lw=2, color="#1f77b4", label="Baseline S21")
    ax.plot(f_ghz[mask], worst_ntwk.s_db[:, 1, 0][mask],
            lw=2, color="#e377c2", label=f"Worst corner #{worst_row['corner']}")
    ax.set_title(f"Baseline vs Worst Corner (S21, k={args.k}σ)")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("S21 (dB)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "worst_corner_overlay.png", dpi=150)
    plt.close(fig)

    logger.info("Done → %s/corner_report.json, corner_table.csv, worst_corner_overlay.png",
                out_dir)


if __name__ == "__main__":
    main()
