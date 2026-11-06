# Energy-Efficient Priority Scheduling for Task Allocation in 6G Edge Computing

Deadline-aware queues and guarded CPU slow-down on top of the EdgePlus scheduler.

**Authors:** Keerat Malhotra, Bhavya Maheshwari, Bhumika Aswal (Netaji Subhas University of Technology, New Delhi)

[EdgePlus](EdgePlus_A_Multiagent_Reinforcement_Learning_Framework_for_Dynamic_Task_Allocation_in_6G_Edge_Computing.pdf)
(Taher et al., *IEEE Internet of Things Journal*, 2026) uses reinforcement learning to pick which edge server
runs each task. Once a task reaches a server, though, EdgePlus queues it first come, first served and runs it at
full CPU speed. This repository rebuilds EdgePlus in a discrete-event simulator and adds two fixed rules inside
every server:

1. **Earliest deadline first, with dropping.** The server runs the waiting task with the closest deadline and
   drops a task as soon as it can no longer finish in time, even at full speed (dropped tasks count as misses).
   EdgePlus's feasibility mask estimates the wait with the same queue order.
2. **Guarded CPU slow-down (DVFS).** A task runs at a lower CPU frequency only when nothing else is waiting for
   that server, the slow-down uses at most 80% of the task's slack, and it adds at most 1.5 s.
   Compute energy is `cycles × kappa × (0.75 f² + 0.25 / f)`, which equals EdgePlus's `kappa × cycles` at full speed.

Neither rule is learned, so both are applied to the **same trained EdgePlus policy**. Every comparison uses
identical policy weights and identical test episodes; only the server rules change.

- Paper: [`paper/main.pdf`](paper/main.pdf) (29 pages, written for readers new to the field), source in
  [`paper/main.tex`](paper/main.tex)
- All numbers with confidence intervals: [`results/tables.md`](results/tables.md)

## Contents

- [Main results](#main-results)
- [Quick start](#quick-start)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Building the paper](#building-the-paper)
- [What is implemented](#what-is-implemented)
- [Datasets](#datasets)
- [Repository layout](#repository-layout)

## Main results

Each row averages 3 training seeds × 10 test episodes. All differences in on-time rate and energy are
significant (Wilcoxon signed-rank test, Benjamini-Hochberg correction, adjusted p < 0.001). The one decrease,
Brasov at 50k, is 0.002 points, about 2 tasks in 100,000.

| Dataset | Load | On time, EdgePlus | On time, ours | URLLC missed | Energy per task (mJ) | 95th-pct latency (s) |
|---|---|---|---|---|---|---|
| Brasov | 50k | 100.00% | 100.00% | 0.00% → 0.01% | 27.04 → 21.69 (−19.8%) | 7.4 → 8.2 |
| Brasov | 100k | 99.87% | 99.99% | 0.66% → 0.04% | 27.61 → 24.86 (−10.0%) | 11.8 → 10.5 |
| Brasov | 150k | 80.52% | **99.65%** | 55.4% → 0.07% | 28.36 → 26.88 (−5.2%) | 148 → 298 |
| Delhi (zero-shot) | 50k | 98.75% | 99.84% | 5.8% → 0.17% | 24.03 → 21.33 (−11.2%) | 25 → 28 |
| Delhi (zero-shot) | 100k | 89.26% | 98.78% | 38.6% → 0.97% | 24.60 → 23.24 (−5.5%) | 110 → 155 |
| Delhi (zero-shot) | 150k | 72.97% | **96.47%** | 73.0% → 3.7% | 25.04 → 23.21 (−7.3%) | 339 → 316 (n.s.) |

"Zero-shot" means the policies were trained on Brasov and evaluated on Delhi without retraining.

**Trade-off.** Overall 95th-percentile latency rises at high load because mMTC tasks (120–600 s deadlines) are
deliberately served later; they still meet their deadlines. URLLC 95th-percentile latency at Brasov 150k drops
from 92 s to 6 s.

**Ablations (Brasov 150k).** Every partial version does worse than the combination:

| Variant | On time |
|---|---|
| EdgePlus baseline (FCFS, keep late tasks) | 80.5% |
| Drop late tasks only (FCFS) | 91.2% |
| Deadline first only (no dropping) | 97.4% |
| Fixed class priority + drop | 96.5% (10.1% of mMTC late) |
| Unguarded DVFS | 46.0% |
| **Both rules** | **99.65%** |

**Fidelity to the original paper (Brasov 150k).** Our EdgePlus reaches 80.5% on time against 82.8% reported
(82.0% for EdgePlus without PSO), with 95th-percentile latency of 148 s against 147 s.

## Quick start

Requires Python 3.11 or newer. Everything runs on CPU.

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy pandas matplotlib
pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m edgeplus.data --dataset all     # downloads ~310 MB, builds data/brasov.npz and data/delhi.npz
```

The trained policies used in the paper are committed in `runs/*/policy.pt` (about 400 KB each), so you can
skip training and go straight to evaluation:

```bash
python -m edgeplus.evaluate --dataset brasov --runs "runs/brasov_*" --episodes 10
python -m edgeplus.analyze                # writes tables, statistics and figures to results/
```

## Reproducing the experiments

```bash
python -m edgeplus.calibrate --dataset brasov            # sanity check with a routing heuristic (no learning)
bash scripts/train_all.sh                                 # 9 training runs (3 loads × 3 seeds)
python -m edgeplus.evaluate --dataset brasov --runs "runs/brasov_*" --episodes 10
python -m edgeplus.evaluate --dataset delhi  --runs "runs/brasov_*" --episodes 10 \
       --variants baseline,edf_drop,guarded_dvfs,full
python -m edgeplus.evaluate --dataset brasov --heuristic-loads 50000,100000,150000 --variants baseline,full
python -m edgeplus.analyze                                # tables, statistics and figures in results/
python -m edgeplus.figures                                # extra figures for the paper, in paper/figures/
```

### Rule variants

Pass these names to `--variants` (default: all). They are defined in `edgeplus/config.py`.

| Name | Queue order | Late tasks | CPU speed |
|---|---|---|---|
| `baseline` | FCFS | kept | full speed (EdgePlus as published) |
| `fcfs_drop` | FCFS | dropped | full speed |
| `edf_keep` | EDF | kept | full speed |
| `class_priority` | URLLC > eMBB > mMTC | dropped | full speed |
| `edf_drop` | EDF | dropped | full speed (Rule 1) |
| `naive_dvfs` | FCFS | kept | slowed whenever there is slack |
| `guarded_dvfs` | FCFS | kept | guarded DVFS (Rule 2) |
| `full` | EDF | dropped | guarded DVFS (both rules) |

### Training time and hardware

Training is CPU-only. A GPU does not help because the policy is a small network and the simulator is the slow
part. Each run uses one core and about 1 GB of RAM. Measured single-core time for 60 PPO updates is about
25 min at 50k, 45 min at 100k and 61 min at 150k: 6.6 CPU-hours for all 9 runs, or about 1.3 hours on an
8-core machine.

`scripts/train_all.sh` runs several jobs in parallel (limit with `JOBS=4 bash scripts/train_all.sh`) and resumes
interrupted runs from their last checkpoint, so it also works on Kaggle or Colab sessions that stop after a few
hours. `DATASET`, `LOADS` and `SEEDS` can be overridden the same way. Single runs and config overrides:

```bash
python -m edgeplus.train --dataset brasov --load 150000 --seed 0 --set ppo.lr=1e-4
```

All parameters live in [`configs/default.toml`](configs/default.toml); values taken from EdgePlus are marked.

## Building the paper

The paper uses the Springer Nature journal template (`sn-jnl`, version 3.1, December 2024; the original package
is the `.zip` in the repository root). The class file `sn-jnl.cls` and the bibliography style
`sn-mathphys-num.bst` (numbered references) are copied into `paper/`, so it builds without extra setup:

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

**Overleaf:** download [`paper-overleaf.zip`](paper-overleaf.zip) from the repository root, then in Overleaf
choose *New Project → Upload Project* and select the zip. It contains `main.tex`, `refs.bib`, `sn-jnl.cls`,
`sn-mathphys-num.bst` and `figures/`, and compiles with Overleaf's default pdfLaTeX compiler. If you change files
in `paper/`, rebuild the zip with:

```bash
cd paper && rm -f ../paper-overleaf.zip && zip -r ../paper-overleaf.zip main.tex refs.bib sn-jnl.cls sn-mathphys-num.bst figures
```

It also builds with `tectonic main.tex`. Useful class options in the first line of `main.tex`:

- `referee`: double line spacing for peer review
- `lineno`: line numbers in the margin
- `sn-mathphys-num` can be swapped for another style from the template's `bst/` folder (for example
  `sn-vancouver-num`) if the target journal asks for it; copy that `.bst` into `paper/` as well.

Keerat Malhotra is marked as the corresponding author (`\author*`); move the `*` to change this. Before
submission, fill in the Declarations block, which is left commented out at the end of `main.tex`.

## What is implemented

| EdgePlus component | Here |
|---|---|
| Masked cooperative PPO with centralized critic (Sec. III-E) | `edgeplus/policy.py`, `edgeplus/train.py`; one shared actor scores every candidate server, the critic sees the whole city |
| Routing order, slice budgets, deadline/memory/battery masks (Sec. III-C, IV-C) | `edgeplus/sim.py` |
| Advisory layer: Nash sharing, PID (gains 1.90/1.20/0.06), leader (Sec. III-F) | `edgeplus/sim.py` (`_tick`, `_nash`) |
| Ledger reputation and debt in state, reward and masks (Sec. III-H) | `edgeplus/sim.py`; PoA validation (0.39 ms) is off the task path and not simulated |
| Reward Eq. 7 + Eq. 15 | `edgeplus/sim.py` (`step`, `_outcome`) |
| Workload labelling (Algorithm 3), class mix 20/50/30, deadlines | `edgeplus/workload.py` |
| PSO outer loop | not implemented (our baseline corresponds to the paper's ablation A2) |
| CBS-D* Lite path planning | replaced by straight-line distance × 1.3 detour factor |
| Cryptography, DKMS, BFT, security and PID robustness experiments | out of scope |

Other modelling choices, applied identically to the baseline and to our rules:

- Servers are multi-core nodes (4 stationary nodes × 24 cores and 1 mobile node × 16 cores per cluster, plus a
  32-core cloud with 2 s extra delay), simulated event by event. The core count was calibrated so that the
  trained EdgePlus model lands close to the paper at 150k (80.5% vs 82.0% on time) and at 100% on time at 50k.
- Each reading becomes one task. An episode takes N consecutive readings and compresses their real time span
  into 600 s. Training uses the first 70% of Brasov's days; testing uses the rest.
- The mask estimates the wait from the server's real backlog instead of the paper's Kingman formula.
- Brasov readings are generated data (CityPulse describes them as a bounded random walk) and arrive at a flat
  rate, so Brasov at 100k is nearly congestion-free (99.9% on time), unlike the paper's 82.5%. AirDelhi is real
  measured data with bursty arrivals; readings taken while a bus is parked are removed (31% of readings).

## Datasets

- **CityPulse pollution data**, which EdgePlus calls Brasov (449 sites, Aug–Oct 2014):
  [CityPulse dataset page](http://iot.ee.surrey.ac.uk:8080/datasets.html). The download file is named
  `citypulse_pollution_csv_data_aarhus_aug_oct_2014.tar.gz`.
- **AirDelhi** (Chauhan et al., NeurIPS 2023 Datasets and Benchmarks, CC BY 4.0):
  [Hugging Face](https://huggingface.co/datasets/sachin-iitd/DelhiPollDataset). Low-cost PM sensors on 13 public
  buses; we use Dec 2020 for training and Jan 2021 for test episodes.

`python -m edgeplus.data` downloads both and caches them in `data/` (ignored by git).

## Repository layout

```
configs/default.toml     all parameters (EdgePlus values are marked)
edgeplus/
  config.py              config loading and the rule variants
  data.py                download and cache both datasets
  workload.py            readings -> tasks, episodes, weighted K-means topology
  sim.py                 discrete-event simulator, masks, advisory layer, server rules
  policy.py              actor-critic network and fast NumPy inference
  train.py               PPO training (resumable)
  evaluate.py            run frozen policies under each rule variant
  analyze.py             tables, confidence intervals, paired tests, figures
  calibrate.py           simulator check with a routing heuristic
  figures.py             explanatory figures for the paper
scripts/train_all.sh     train all loads and seeds
runs/                    trained policies, training logs and run configs
results/                 evaluation CSVs, tables.md, summary.json, paired tests, figures
paper/                   main.tex (sn-jnl template), refs.bib, figures/, compiled main.pdf
```

The repository root also holds the EdgePlus paper, a cited Journal of Cloud Computing article
(`s13677-026-00857-x_*.pdf`) and the Springer Nature template package.
