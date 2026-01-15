import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import skrf as rf

from mc_worst_case import perturb_network_with_params, compute_scalar_objective


def run_mc_summary(base: rf.Network, seed: int, n: int, fmin, fmax, objective,
                   loss_sigma_db, slope_sigma_db, refl_sigma_db):
    rng = np.random.default_rng(seed)
    f_ghz = base.f / 1e9
    objs = np.zeros(n, dtype=float)

    worst_obj = float("inf")
    worst_params = None

    for i in range(n):
        nt, params = perturb_network_with_params(
            base, rng,
            loss_mu_db=0.0, loss_sigma_db=loss_sigma_db,
            slope_sigma_db=slope_sigma_db,
            refl_mu_db=0.0, refl_sigma_db=refl_sigma_db,
        )
        obj = compute_scalar_objective(f_ghz, nt.s_db[:, 1, 0], fmin, fmax, objective)
        objs[i] = obj
        if obj < worst_obj:
            worst_obj = obj
            worst_params = params

    return {
        "mean": float(np.mean(objs)),
        "var": float(np.var(objs, ddof=1)) if n >= 2 else float("nan"),
        "p01": float(np.percentile(objs, 1)),
        "p05": float(np.percentile(objs, 5)),
        "p50": float(np.percentile(objs, 50)),
        "p95": float(np.percentile(objs, 95)),
        "p99": float(np.percentile(objs, 99)),
        "worst": float(np.min(objs)),
        "worst_params": worst_params,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2p", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n", type=int, default=500, help="MC samples per DOE point (keep small)")
    ap.add_argument("--fmin", type=float, default=None)
    ap.add_argument("--fmax", type=float, default=None)
    ap.add_argument("--objective", default="s21_min", choices=["s21_min", "s21_mean"])

    # two-level DOE settings (low/high)
    ap.add_argument("--loss-sigma-low", type=float, default=0.3)
    ap.add_argument("--loss-sigma-high", type=float, default=0.7)
    ap.add_argument("--slope-sigma-low", type=float, default=0.1)
    ap.add_argument("--slope-sigma-high", type=float, default=0.3)
    ap.add_argument("--refl-sigma-low", type=float, default=0.3)
    ap.add_argument("--refl-sigma-high", type=float, default=0.7)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = rf.Network(args.s2p)
    levels = {
        "loss_sigma_db": [args.loss_sigma_low, args.loss_sigma_high],
        "slope_sigma_db": [args.slope_sigma_low, args.slope_sigma_high],
        "refl_sigma_db": [args.refl_sigma_low, args.refl_sigma_high],
    }

    keys = list(levels.keys())
    combos = list(itertools.product(*[levels[k] for k in keys]))

    rows = []
    for i, (ls, ss, rs) in enumerate(combos):
        summary = run_mc_summary(
            base=base,
            seed=args.seed + i,
            n=args.n,
            fmin=args.fmin,
            fmax=args.fmax,
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

    with (out_dir / "doe_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    report = {
        "input": args.s2p,
        "objective": args.objective,
        "band_ghz": {"fmin": args.fmin, "fmax": args.fmax},
        "n_per_point": args.n,
        "levels": levels,
        "notes": "2-level full factorial DOE over sigma parameters; response=MC distribution stats of objective",
    }
    (out_dir / "doe_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[OK] Saved: {out_dir}/doe_results.csv, doe_report.json")


if __name__ == "__main__":
    main()
