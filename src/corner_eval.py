import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import skrf as rf


def apply_corner(ntwk: rf.Network, a_loss_db: float, b_slope_db: float, refl_db: float) -> rf.Network:
    f = ntwk.f.astype(float)
    f_norm = (f - f.min()) / (f.max() - f.min() + 1e-12)
    extra_loss_db = a_loss_db + b_slope_db * (f_norm - 0.5)

    s = ntwk.s.copy()

    # S21
    s21 = s[:, 1, 0]
    mag21 = np.abs(s21)
    ph21 = np.angle(s21)
    mag21_new = mag21 * (10 ** (-extra_loss_db / 20.0))
    s[:, 1, 0] = mag21_new * np.exp(1j * ph21)

    # S11
    s11 = s[:, 0, 0]
    mag11 = np.abs(s11)
    ph11 = np.angle(s11)
    mag11_new = mag11 * (10 ** (refl_db / 20.0))
    mag11_new = np.minimum(mag11_new, 0.999999)
    s[:, 0, 0] = mag11_new * np.exp(1j * ph11)

    out = ntwk.copy()
    out.s = s
    return out


def objective_s21(f_ghz, s21_db, fmin=None, fmax=None, mode="s21_min") -> float:
    mask = np.ones_like(f_ghz, dtype=bool)
    if fmin is not None:
        mask &= (f_ghz >= fmin)
    if fmax is not None:
        mask &= (f_ghz <= fmax)

    x = s21_db[mask]
    if len(x) == 0:
        raise ValueError("No points in selected band.")
    if mode == "s21_min":
        return float(np.min(x))
    if mode == "s21_mean":
        return float(np.mean(x))
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2p", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fmin", type=float, default=None)
    ap.add_argument("--fmax", type=float, default=None)
    ap.add_argument("--objective", default="s21_min", choices=["s21_min", "s21_mean"])

    # corner strength: use +/- k*sigma as corner magnitude
    ap.add_argument("--k", type=float, default=3.0)
    ap.add_argument("--loss-sigma-db", type=float, default=0.5)
    ap.add_argument("--slope-sigma-db", type=float, default=0.2)
    ap.add_argument("--refl-sigma-db", type=float, default=0.5)

    # optional: compare with MC report
    ap.add_argument("--mc-report", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = rf.Network(args.s2p)
    f_ghz = base.f / 1e9
    base_obj = objective_s21(f_ghz, base.s_db[:, 1, 0], args.fmin, args.fmax, args.objective)

    # 8 corners: sign combinations for (a, b, refl)
    corners = []
    for sa in (-1, 1):
        for sb in (-1, 1):
            for sr in (-1, 1):
                a = sa * args.k * args.loss_sigma_db
                b = sb * args.k * args.slope_sigma_db
                r = sr * args.k * args.refl_sigma_db
                corners.append((a, b, r))

    rows = []
    worst_obj = float("inf")
    worst_cfg = None
    worst_ntwk = None

    for idx, (a, b, r) in enumerate(corners):
        nt = apply_corner(base, a, b, r)
        obj = objective_s21(f_ghz, nt.s_db[:, 1, 0], args.fmin, args.fmax, args.objective)
        rows.append({"corner": idx, "a_loss_db": a, "b_slope_db": b, "refl_db": r, "objective_db": obj})
        if obj < worst_obj:
            worst_obj = obj
            worst_cfg = rows[-1]
            worst_ntwk = nt

    # write table
    with (out_dir / "corner_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["corner", "a_loss_db", "b_slope_db", "refl_db", "objective_db"])
        w.writeheader()
        w.writerows(rows)

    report = {
        "input": args.s2p,
        "band_ghz": {"fmin": args.fmin, "fmax": args.fmax},
        "objective": args.objective,
        "baseline_obj": base_obj,
        "k": args.k,
        "sigmas": {"loss_sigma_db": args.loss_sigma_db, "slope_sigma_db": args.slope_sigma_db, "refl_sigma_db": args.refl_sigma_db},
        "corner_worst": worst_cfg,
    }

    if args.mc_report:
        mc = json.loads(Path(args.mc_report).read_text(encoding="utf-8"))
        mc_worst = float(mc["obj_stats"]["worst"])
        report["mc_worst"] = mc_worst
        report["miss_gap_db"] = float(worst_obj - mc_worst)  # >0 means corner didn't reach MC worst (miss)

    (out_dir / "corner_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # overlay baseline vs worst-corner
    mask = np.ones_like(f_ghz, dtype=bool)
    if args.fmin is not None:
        mask &= (f_ghz >= args.fmin)
    if args.fmax is not None:
        mask &= (f_ghz <= args.fmax)

    plt.figure(figsize=(10, 4))
    plt.plot(f_ghz[mask], base.s_db[:, 1, 0][mask], linewidth=2, label="baseline S21")
    plt.plot(f_ghz[mask], worst_ntwk.s_db[:, 1, 0][mask], linewidth=2, label="worst-corner S21")
    plt.title("Baseline vs worst-corner (S21)")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("S21 (dB)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "worst_corner_overlay.png", dpi=150)

    print(f"[OK] Saved: {out_dir}/corner_report.json, corner_table.csv, worst_corner_overlay.png")


if __name__ == "__main__":
    main()
