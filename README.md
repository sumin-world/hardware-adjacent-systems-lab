# Channel Analysis Lab

**Statistical Worst-Case Analysis for High-Speed Serial Channel Characterization**

[![CI](https://github.com/YOUR_USERNAME/channel-analysis-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/channel-analysis-lab/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Motivation

Modern high-speed SerDes links (PCIe 6.0, USB4, 112G PAM4) demand rigorous channel qualification beyond simple pass/fail. Manufacturing variation in PCB trace geometry, dielectric properties, and connector tolerances creates a distribution of channel performance — and the tail matters most.

This toolkit ingests Touchstone S-parameter data and applies industry-standard statistical methods to answer:

- **How bad can the channel get?** → Monte Carlo worst-case extraction with configurable perturbation models
- **Do deterministic corners cover the statistical tail?** → Corner vs. MC gap analysis
- **Which variation parameters dominate performance?** → DOE sensitivity ranking

These techniques align with methodologies used in channel compliance frameworks at companies like Qualcomm (QSiP channel modeling), NVIDIA (NVLink/PCIe SI validation), Intel (HSIO), and Samsung (HBM PHY characterization).

## Architecture

```
┌─────────────┐     ┌────────────────────┐     ┌──────────────────┐
│  .s2p file   │────▶│   analyze_s2p.py   │────▶│  Baseline metrics │
│ (Touchstone) │     │  S21/S11 extraction │     │  + S-param plots  │
└──────┬───────┘     └────────────────────┘     └──────────────────┘
       │
       ├─────────────┐
       │             ▼
       │     ┌────────────────────┐     ┌──────────────────────────┐
       │     │  mc_worst_case.py  │────▶│  Objective distribution   │
       │     │  Vectorized MC     │     │  Worst-case overlay       │
       │     │  (N=2000+ samples) │     │  Percentile report (JSON) │
       │     └────────┬───────────┘     └──────────────────────────┘
       │              │
       │              │ mc_report.json
       │              ▼
       │     ┌────────────────────┐     ┌──────────────────────────┐
       │     │   corner_eval.py   │────▶│  Corner table (CSV)       │
       │     │  2^3 deterministic │     │  Corner vs MC gap (dB)    │
       │     │  corner sweep      │     │  Worst-corner overlay     │
       │     └────────────────────┘     └──────────────────────────┘
       │
       │     ┌────────────────────┐     ┌──────────────────────────┐
       └────▶│   doe_runner.py    │────▶│  DOE results (CSV)        │
             │  2^k full factorial │     │  Sensitivity main effects │
             │  over σ parameters │     │  Pareto chart             │
             └────────────────────┘     └──────────────────────────┘
```

## Key Features

| Feature | Description |
|---|---|
| **Vectorized Monte Carlo** | Fully NumPy-vectorized perturbation engine — no Python loop over samples. 10–50× faster than naive per-sample iteration for large N. |
| **Perturbation Model** | 3-parameter Gaussian model: flat loss (`a`), frequency-dependent slope (`b`), reflection degradation (`refl`). Captures dominant manufacturing variations in PCB channels. |
| **Corner Coverage Analysis** | Evaluates whether 2^3 deterministic corners (±k·σ) envelop the MC statistical tail. Reports gap in dB if corners miss. |
| **DOE Sensitivity** | 2-level full factorial over σ parameters. Computes main effects to identify which variation source dominates worst-case performance. |
| **Pluggable Objectives** | `s21_min` (worst-point insertion loss) or `s21_mean` (average IL over band). Extensible to custom metrics. |
| **CI-Ready** | GitHub Actions workflow with compile check, smoke tests, and artifact validation. |

## Quick Start

### Prerequisites

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Generate Sample Data (for demo)

```bash
python -c "
import pathlib, skrf as rf
pathlib.Path('data/real').mkdir(parents=True, exist_ok=True)
rf.data.ring_slot.write_touchstone('data/real/sample')
print('Wrote: data/real/sample.s2p')
"
```

### Run Full Pipeline

```bash
# 1. Baseline S-parameter analysis
python src/analyze_s2p.py --s2p data/real/sample.s2p --out outputs/analyze --fmin 75 --fmax 110

# 2. Monte Carlo worst-case (vectorized, N=2000)
python src/mc_worst_case.py --s2p data/real/sample.s2p --out outputs/mc --n 2000 --seed 42 --fmin 75 --fmax 110

# 3. Corner evaluation (compare corners vs MC tail)
python src/corner_eval.py --s2p data/real/sample.s2p --out outputs/corner --fmin 75 --fmax 110 --mc-report outputs/mc/mc_report.json

# 4. DOE sensitivity analysis
python src/doe_runner.py --s2p data/real/sample.s2p --out outputs/doe --n 500 --fmin 75 --fmax 110
```

Or use the Makefile:

```bash
make setup sample demo
```

## Perturbation Model

Each Monte Carlo trial applies three independent Gaussian perturbations to the baseline S-parameters:

| Parameter | Symbol | Default σ | Physical Meaning |
|---|---|---|---|
| Flat loss | `a` | 0.5 dB | Conductor/dielectric loss variation (e.g., Dk/Df tolerance) |
| Slope | `b` | 0.2 dB | Frequency-dependent loss tilt (skin effect variation, roughness) |
| Reflection | `refl` | 0.5 dB | Impedance mismatch degradation (trace width/spacing tolerance) |

The perturbed S-parameters are computed as:

```
S21_new(f) = S21_base(f) · 10^(-(a + b·(f_norm - 0.5)) / 20)
S11_new(f) = min(|S11_base(f)| · 10^(refl / 20), 0.999999) · e^(j·∠S11)
```

## Output Artifacts

| Script | Outputs | Description |
|---|---|---|
| `analyze_s2p.py` | `metrics.json`, `sparams.png` | Baseline S21/S11 statistics and frequency-domain plots |
| `mc_worst_case.py` | `mc_report.json`, `mc_objectives.csv`, `mc_hist.png`, `worst_overlay.png` | MC distribution, percentiles (p01/p05/p50/p95/p99), worst-case S21 overlay |
| `corner_eval.py` | `corner_report.json`, `corner_table.csv`, `worst_corner_overlay.png` | 8-corner sweep results, gap vs MC worst |
| `doe_runner.py` | `doe_report.json`, `doe_results.csv`, `doe_sensitivity.png` | Full factorial results, main effect Pareto chart |

## Sample Results

### Monte Carlo Objective Distribution
![MC Histogram](docs/img/mc_hist.png)

### Baseline vs Worst-Case Overlay
![Worst Overlay](docs/img/worst_overlay.png)

### DOE Sensitivity (Main Effects)
![DOE Pareto](docs/img/doe_sensitivity.png)

## Project Structure

```
channel-analysis-lab/
├── src/
│   ├── common.py           # Shared utilities (perturbation engine, objectives, I/O)
│   ├── analyze_s2p.py      # Baseline S-parameter analysis
│   ├── mc_worst_case.py    # Vectorized Monte Carlo worst-case
│   ├── corner_eval.py      # Deterministic corner sweep
│   └── doe_runner.py       # 2^k full factorial DOE
├── data/real/              # User-supplied .s2p files (gitignored)
├── outputs/                # Generated reports and plots (gitignored)
├── docs/img/               # README screenshots
├── .github/workflows/ci.yml
├── requirements.txt
├── Makefile
├── pyproject.toml
└── LICENSE
```

## Extending

**Custom objective functions**: Add a new mode to `compute_scalar_objective()` in `common.py`. For example, integrated insertion loss over a compliance mask:

```python
if mode == "il_integrated":
    return float(np.trapz(np.abs(x), f_ghz[mask]))
```

**Additional perturbation parameters**: Extend `perturb_network_vectorized()` to add crosstalk coupling, via stub resonance, or connector mismatch terms.

**Real-world data**: Replace `data/real/sample.s2p` with measured Touchstone files from VNA exports or EM simulation (HFSS, CST, ADS).

## Industry Context

This type of statistical channel analysis is central to:

- **PCIe 5.0/6.0 channel compliance** — CEM spec requires insertion loss, return loss, and crosstalk budgets with statistical margin
- **112G PAM4 SerDes design** — COM (Channel Operating Margin) methodology uses similar MC perturbation over channel segments
- **HBM3/3e PHY validation** — Memory interface SI requires worst-case eye height estimation across PVT corners
- **Qualcomm QSiP / NVIDIA NVLink** — Multi-chip package channels validated with statistical loss models + corner analysis

## License

MIT — see [LICENSE](LICENSE).
