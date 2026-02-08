#!/usr/bin/env python3
"""
analyze_s2p.py — Baseline S-parameter extraction and visualization.

Reads a 2-port Touchstone (.s2p) file, computes S21/S11 statistics over a
user-specified frequency band, and generates publication-quality plots.

Outputs:
    metrics.json   — S21/S11 percentile statistics, worst-case point
    sparams.png    — Transmission and reflection frequency-domain plots
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import band_mask, load_network, percentile_stats, write_json

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.grid": True,
    "axes.grid.which": "both",
    "grid.alpha": 0.3,
    "font.size": 11,
})


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Baseline S-parameter analysis for a 2-port Touchstone file.",
    )
    ap.add_argument("--s2p", required=True, help="Path to .s2p file")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--fmin", type=float, default=None, help="Band lower bound (GHz)")
    ap.add_argument("--fmax", type=float, default=None, help="Band upper bound (GHz)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ntwk = load_network(args.s2p)

    f_ghz = ntwk.f / 1e9
    s21_db = ntwk.s_db[:, 1, 0]
    s11_db = ntwk.s_db[:, 0, 0]

    mask = band_mask(f_ghz, args.fmin, args.fmax)
    f_b = f_ghz[mask]
    s21_b = s21_db[mask]
    s11_b = s11_db[mask]

    # ── Report ────────────────────────────────────────────────────────────
    worst_idx = int(np.argmin(s21_b))
    report = {
        "input": str(args.s2p),
        "freq_ghz": {
            "min": float(f_b[0]),
            "max": float(f_b[-1]),
            "points": int(len(f_b)),
        },
        "metrics": {
            "s21_db": percentile_stats(s21_b),
            "s11_db": percentile_stats(s11_b),
        },
        "worst_case": {
            "s21_db_min": float(s21_b[worst_idx]),
            "at_freq_ghz": float(f_b[worst_idx]),
        },
    }
    write_json(out_dir / "metrics.json", report)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(f_b, s21_b, linewidth=1.8, color="#1f77b4", label="S21")
    axes[0].axhline(y=s21_b[worst_idx], ls="--", lw=1, color="#d62728",
                     label=f"worst = {s21_b[worst_idx]:.2f} dB")
    axes[0].set_ylabel("Magnitude (dB)")
    axes[0].set_title("Insertion Loss (S21)")
    axes[0].legend(loc="lower left")

    axes[1].plot(f_b, s11_b, linewidth=1.8, color="#ff7f0e", label="S11")
    axes[1].set_ylabel("Magnitude (dB)")
    axes[1].set_xlabel("Frequency (GHz)")
    axes[1].set_title("Return Loss (S11)")
    axes[1].legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(out_dir / "sparams.png", dpi=150)
    plt.close(fig)

    logger.info("Done → %s/metrics.json, sparams.png", out_dir)


if __name__ == "__main__":
    main()
