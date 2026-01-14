import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import skrf as rf


def compute_stats(x: np.ndarray) -> dict:
    return {
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "var": float(np.var(x, ddof=1)) if len(x) >= 2 else float("nan"),
        "p01": float(np.percentile(x, 1)),
        "p05": float(np.percentile(x, 5)),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2p", required=True, help="Path to .s2p (Touchstone) file")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--fmin", type=float, default=None, help="Min freq (GHz) for band")
    ap.add_argument("--fmax", type=float, default=None, help="Max freq (GHz) for band")
    args = ap.parse_args()

    s2p_path = Path(args.s2p)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ntwk = rf.Network(str(s2p_path))
    if ntwk.nports != 2:
        raise ValueError(f"Expected 2-port network, got {ntwk.nports}-port")

    f_ghz = ntwk.f / 1e9
    s21_db = ntwk.s_db[:, 1, 0]
    s11_db = ntwk.s_db[:, 0, 0]

    mask = np.ones_like(f_ghz, dtype=bool)
    if args.fmin is not None:
        mask &= (f_ghz >= args.fmin)
    if args.fmax is not None:
        mask &= (f_ghz <= args.fmax)

    f_ghz_b = f_ghz[mask]
    s21_db_b = s21_db[mask]
    s11_db_b = s11_db[mask]

    if len(f_ghz_b) == 0:
        raise ValueError("No points in selected band. Check --fmin/--fmax and file frequency range.")

    report = {
        "input": str(s2p_path),
        "freq_ghz": {"min": float(f_ghz_b[0]), "max": float(f_ghz_b[-1]), "points": int(len(f_ghz_b))},
        "metrics": {
            "s21_db": compute_stats(s21_db_b),
            "s11_db": compute_stats(s11_db_b),
        },
        "worst_case": {
            "s21_db_min": float(np.min(s21_db_b)),
            "at_freq_ghz": float(f_ghz_b[int(np.argmin(s21_db_b))]),
        },
    }

    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(f_ghz_b, s21_db_b, linewidth=2, label="S21 (dB)")
    ax1.set_title("Transmission (S21)")
    ax1.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(f_ghz_b, s11_db_b, linewidth=2, label="S11 (dB)")
    ax2.set_title("Reflection (S11)")
    ax2.set_xlabel("Frequency (GHz)")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "sparams.png", dpi=150)
    print(f"[OK] Saved: {out_dir}/metrics.json, {out_dir}/sparams.png")


if __name__ == "__main__":
    main()
