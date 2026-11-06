# Slack-aware server rules for EdgePlus

This repo reimplements **EdgePlus** (Taher et al., *IEEE Internet of Things Journal*, 2026; PDF in the repo
root) and adds two rules that run on each edge server:

1. **Deadline-first queueing with dropping of late tasks.** A server runs the waiting task with the earliest
   deadline first (EDF) instead of first come first served, and drops a task once it can no longer meet its
   deadline even at full CPU speed (dropped tasks count as failures). EdgePlus's feasibility mask estimates
   the wait using the same queue order.
2. **Guarded CPU slow-down (DVFS).** A task runs at a lower CPU frequency only when nothing else is waiting
   for that server, the slow-down uses at most 80% of the task's slack, and it adds at most 1.5 s.
   Compute energy is `cycles x kappa x (0.75 f^2 + 0.25 / f)`, which equals EdgePlus's `kappa x cycles`
   at full speed.

Both are fixed rules, so they are applied to the **same trained EdgePlus policy**: every comparison below uses
identical policy weights and identical test episodes, and only the server rules change.

The paper draft is in [`paper/`](paper/) and all numbers are in [`results/tables.md`](results/tables.md).

## Main results

Trained EdgePlus policy, 3 training seeds x 10 test episodes per row, mean values
(95% confidence intervals are in `results/tables.md`; all differences in on-time rate and energy are
significant, Wilcoxon signed-rank with Benjamini-Hochberg correction, adjusted p < 0.001; at Brasov 50k
the on-time rate is 0.002 points lower, about 2 tasks in 100,000).

| Dataset | Load | On-time, EdgePlus | On-time, ours | URLLC missed, EdgePlus → ours | Energy per task (mJ), EdgePlus → ours | 95th-pct latency (s), EdgePlus → ours |
|---|---|---|---|---|---|---|
| Brasov | 50k | 100.00% | 100.00% | 0.00% → 0.01% | 27.04 → 21.69 (−19.8%) | 7.4 → 8.2 |
| Brasov | 100k | 99.87% | 99.99% | 0.66% → 0.04% | 27.61 → 24.86 (−10.0%) | 11.8 → 10.5 |
| Brasov | 150k | 80.52% | 99.65% | 55.4% → 0.07% | 28.36 → 26.88 (−5.2%) | 148 → 298 |
| Delhi (zero-shot) | 50k | 98.75% | 99.84% | 5.8% → 0.17% | 24.03 → 21.33 (−11.2%) | 25 → 28 |
| Delhi (zero-shot) | 100k | 89.26% | 98.78% | 38.6% → 0.97% | 24.60 → 23.24 (−5.5%) | 110 → 155 |
| Delhi (zero-shot) | 150k | 72.97% | 96.47% | 73.0% → 3.7% | 25.04 → 23.21 (−7.3%) | 339 → 316 (n.s.) |

The trade-off: overall 95th-percentile latency gets worse at high load because mMTC tasks (120-600 s
deadlines) are deliberately served later. URLLC's 95th-percentile latency at Brasov 150k drops from 92 s to
6 s. In the ablations, every partial version does worse than the combination: at Brasov 150k, deadline-first
without dropping reaches 97.4% on time, dropping without deadline-first 91.2%, fixed class priority 96.5%
(with 10.1% of mMTC tasks late), and unguarded DVFS 46.0% (see `results/tables.md`).

Fidelity check against the paper (Brasov, 150k): our EdgePlus reaches 80.5% on time vs 82.8% reported
(82.0% for EdgePlus without PSO), with 95th-percentile latency 148 s vs 147 s.

## Setup

Python 3.11 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy pandas matplotlib
pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m edgeplus.data --dataset all      # downloads ~310 MB, builds data/brasov.npz and data/delhi.npz
```

## Reproduce everything

```bash
python -m edgeplus.calibrate --dataset brasov            # sanity check with a simple routing rule (no learning)
bash scripts/train_all.sh                                 # 9 training runs (3 loads x 3 seeds)
python -m edgeplus.evaluate --dataset brasov --runs "runs/brasov_*" --episodes 10
python -m edgeplus.evaluate --dataset delhi  --runs "runs/brasov_*" --episodes 10 \
       --variants baseline,edf_drop,guarded_dvfs,full
python -m edgeplus.evaluate --dataset brasov --heuristic-loads 50000,100000,150000 --variants baseline,full
python -m edgeplus.analyze                                # tables, statistics and figures in results/
```

The trained policies used in the paper are committed in `runs/*/policy.pt` (about 400 KB each), so the
evaluation and analysis steps can be run without training.

### Where and how long to train

Training runs on CPU only; a GPU does not help because the policy is a small network and the simulator is
the slow part. Each run uses one core and about 1 GB of RAM. Measured single-core time for 60 PPO updates:
about 25 min at 50k, 45 min at 100k and 61 min at 150k, or 6.6 CPU-hours for all 9 runs (about 1.3 hours
on an 8-core machine). `scripts/train_all.sh` runs several jobs at once (`JOBS=4 bash scripts/train_all.sh`
to limit it) and resumes interrupted runs from their last checkpoint, so it also works on Kaggle or Colab
sessions that stop after a few hours.

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
| CBS-D* Lite path planning | replaced by straight-line distance x 1.3 detour factor |
| Cryptography, DKMS, BFT, security and PID robustness experiments | out of scope |

Other modelling choices, all applied identically to the baseline and to our rules:
- Servers are multi-core nodes (4 stationary nodes x 24 cores and 1 mobile node x 16 cores per cluster,
  plus a 32-core cloud with 2 s extra delay), simulated event by event. The core count was calibrated so that
  a simple routing rule gives about 100% on time at 50k and about 80% at 150k, as in the paper.
- Each reading becomes one task; an episode takes N consecutive readings and compresses their real time span
  into 600 s. Training uses the first 70% of Brasov's days, testing the rest.
- The mask estimates the wait from the server's real backlog instead of the paper's Kingman formula.
- Brasov readings are generated data (CityPulse describes them as a bounded random walk) and arrive at a flat
  rate, so Brasov at 100k is nearly congestion-free (99.9% on time), unlike the paper's 82.5%. AirDelhi is real
  measured data with bursty arrivals; readings taken while a bus is parked are removed (31% of readings).

## Datasets

- **CityPulse pollution data**, the set EdgePlus calls Brasov (449 sites, Aug-Oct 2014),
  [CityPulse dataset page](http://iot.ee.surrey.ac.uk:8080/datasets.html). The download file is named
  `citypulse_pollution_csv_data_aarhus_aug_oct_2014.tar.gz`.
- **AirDelhi** (Chauhan et al., NeurIPS 2023 Datasets and Benchmarks, CC BY 4.0),
  [Hugging Face](https://huggingface.co/datasets/sachin-iitd/DelhiPollDataset): low-cost PM sensors on
  13 public buses; we use Dec 2020 and Jan 2021 (test episodes come from January).

## Layout

```
configs/default.toml     all parameters (EdgePlus values are marked)
edgeplus/data.py         download and cache both datasets
edgeplus/workload.py     readings -> tasks, episodes, weighted K-means topology
edgeplus/sim.py          discrete-event simulator, masks, advisory layer, server rules
edgeplus/policy.py       actor-critic network and fast NumPy inference
edgeplus/train.py        PPO training (resumable)
edgeplus/evaluate.py     run frozen policies under each rule variant
edgeplus/analyze.py      tables, confidence intervals, paired tests, figures
edgeplus/calibrate.py    simulator check with a routing heuristic
scripts/train_all.sh     train all loads and seeds
runs/                    trained policies and training logs
results/                 evaluation CSVs, tables.md, figures
paper/                   paper draft (IEEE conference format)
```
