import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import skrf as rf


def perturb_network_with_params(
    ntwk: rf.Network,
    rng: np.random.Generator,
    loss_mu_db=0.0,
    loss_sigma_db=0.5,
    slope_sigma_db=0.2,
    refl_mu_db=0.0,
    refl_sigma_db=0.5,
):
    """
    Samples parameters and returns (perturbed_network, params).
    params:
      a_loss_db: scalar
      b_slope_db: scalar (applies across band)
      refl_db: scalar
    """
    f = ntwk.f.astype(float)
    f_norm = (f - f.min()) / (f.max() - f.min() + 1e-12)

    a = float(rng.normal(loss_mu_db, loss_sigma_db))
    b = float(rng.normal(0.0, slope_sigma_db))
    refl = float(rng.normal(refl_mu_db, refl_sigma_db))

    extra_loss_db = a + b * (f_norm - 0.5)  # (N,)

    s = ntwk.s.copy()

    # S21: add loss
    s21 = s[:, 1, 0]
    mag21 = np.abs(s21)
    ph21 = np.angle(s21)
    mag21_new = mag21 * (10 ** (-extra_loss_db / 20.0))
    s[:, 1, 0] = mag21_new * np.exp(1j * ph21)

    # S11: worsen reflection
    s11 = s[:, 0, 0]
    mag11 = np.abs(s11)
    ph11 = np.angle(s11)
    mag11_new = mag11 * (10 ** (refl / 20.0))
    mag11_new = np.minimum(mag11_new, 0.999999)
    s[:, 0, 0] = mag11_new * np.exp(1j * ph11)

    ntwk_new = ntwk.copy()
    ntwk_new.s = s

    params = {"a_loss_db": a, "b_slope_db": b, "refl_db": refl}
    return ntwk_new, params


def compute_scalar_objective(f_ghz, s21_db, fmin=None, fmax=None, mode="s21_min"):
    mask = np.ones_like(f_ghz, dtype=bool)
    if fmin is not None:
        mask &= (f_ghz >= fmin)
    if fmax is not None:
        mask &= (f_ghz <= fmax)

    x = s21_db[mask]
    if len(x) == 0:
        raise ValueError("No points in selected band. Check --fmin/--fmax.")
    if mode == "s21_min":
        return float(np.min(x))
    if mode == "s21_mean":
        return float(np.mean(x))
    raise ValueError(f"Unknown mode: {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2p", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fmin", type=float, default=None)
    ap.add_argument("--fmax", type=float, default=None)
    ap.add_argument("--objective", default="s21_min", choices=["s21_min", "s21_mean"])

    ap.add_argument("--loss-mu-db", type=float, default=0.0)
    ap.add_argument("--loss-sigma-db", type=float, default=0.5)
    ap.add_argument("--slope-sigma-db", type=float, default=0.2)
    ap.add_argument("--refl-mu-db", type=float, default=0.0)
    ap.add_argument("--refl-sigma-db", type=float, default=0.5)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = rf.Network(args.s2p)
    if base.nports != 2:
        raise ValueError(f"Expected 2-port network, got {base.nports}-port")

    rng = np.random.default_rng(args.seed)

    f_ghz = base.f / 1e9
    base_s21_db = base.s_db[:, 1, 0]
    base_obj = compute_scalar_objective(f_ghz, base_s21_db, args.fmin, args.fmax, args.objective)

    objs = np.zeros(args.n, dtype=float)

    worst_obj = float("inf")
    worst_idx = -1
    worst_ntwk = None
    worst_params = None

    for i in range(args.n):
        nt, params = perturb_network_with_params(
            base,
            rng,
            loss_mu_db=args.loss_mu_db,
            loss_sigma_db=args.loss_sigma_db,
            slope_sigma_db=args.slope_sigma_db,
            refl_mu_db=args.refl_mu_db,
            refl_sigma_db=args.refl_sigma_db,
        )
        s21_db = nt.s_db[:, 1, 0]
        obj = compute_scalar_objective(f_ghz, s21_db, args.fmin, args.fmax, args.objective)
        objs[i] = obj

        if obj < worst_obj:
            worst_obj = obj
            worst_idx = i
            worst_ntwk = nt
            worst_params = params

    stats = {
        "input": args.s2p,
        "n": args.n,
        "seed": args.seed,
        "band_ghz": {"fmin": args.fmin, "fmax": args.fmax},
        "objective": args.objective,
        "baseline_obj": base_obj,
        "obj_stats": {
            "mean": float(np.mean(objs)),
            "var": float(np.var(objs, ddof=1)) if len(objs) >= 2 else float("nan"),
            "p01": float(np.percentile(objs, 1)),
            "p05": float(np.percentile(objs, 5)),
            "p50": float(np.percentile(objs, 50)),
            "p95": float(np.percentile(objs, 95)),
            "p99": float(np.percentile(objs, 99)),
            "worst": float(np.min(objs)),
        },
        "worst": {"idx": worst_idx, "obj": worst_obj, "params": worst_params},
        "perturb_params": {
            "loss_mu_db": args.loss_mu_db,
            "loss_sigma_db": args.loss_sigma_db,
            "slope_sigma_db": args.slope_sigma_db,
            "refl_mu_db": args.refl_mu_db,
            "refl_sigma_db": args.refl_sigma_db,
        },
    }
    (out_dir / "mc_report.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    with (out_dir / "mc_objectives.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample", "objective"])
        for i, v in enumerate(objs):
            w.writerow([i, float(v)])

    plt.figure(figsize=(10, 4))
    plt.hist(objs, bins=60)
    plt.title("Monte Carlo objective distribution")
    plt.xlabel("objective (dB)")
    plt.ylabel("count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "mc_hist.png", dpi=150)

    base_s21_db = base.s_db[:, 1, 0]
    worst_s21_db = worst_ntwk.s_db[:, 1, 0]

    mask = np.ones_like(f_ghz, dtype=bool)
    if args.fmin is not None:
        mask &= (f_ghz >= args.fmin)
    if args.fmax is not None:
        mask &= (f_ghz <= args.fmax)

    plt.figure(figsize=(10, 4))
    plt.plot(f_ghz[mask], base_s21_db[mask], linewidth=2, label="baseline S21")
    plt.plot(f_ghz[mask], worst_s21_db[mask], linewidth=2, label="worst-case S21")
    plt.title("Baseline vs worst-case (S21)")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("S21 (dB)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "worst_overlay.png", dpi=150)

    print(f"[OK] Saved: {out_dir}/mc_report.json, mc_objectives.csv, mc_hist.png, worst_overlay.png")


if __name__ == "__main__":
    main()
