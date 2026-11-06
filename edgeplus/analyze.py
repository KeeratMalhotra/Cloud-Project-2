"""Turn evaluation CSVs into tables, statistics and figures for the paper.

    python -m edgeplus.analyze

Reads results/eval_*.csv and runs/*/log.csv. Writes results/tables.md, results/summary.json and
results/figures/*.png. Averages use every (training seed, test episode) pair; 95% confidence
intervals use the t distribution; each variant is compared with the baseline using a paired
Wilcoxon signed-rank test (pairs share the same trained policy and test episode) with
Benjamini-Hochberg correction across all tests of a dataset.
"""
import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .config import ROOT

RES = ROOT / "results"
FIG = RES / "figures"
KEY_METRICS = ["success", "viol_URLLC", "viol_eMBB", "viol_mMTC", "lp50", "lp95", "lp95_URLLC", "lp95_eMBB",
               "lp95_mMTC", "norm_p95", "energy_mj", "energy_per_ontime", "energy_total", "cloud_pct",
               "locality_pct", "drop_pct", "slowed_pct", "jain", "throughput", "decision_ms"]
TEST_METRICS = ["success", "viol_URLLC", "viol_eMBB", "viol_mMTC", "lp95", "lp95_URLLC", "lp95_mMTC", "energy_mj"]
LABELS = {"baseline": "EdgePlus (baseline)", "edf_drop": "+ deadline-first & drop", "guarded_dvfs": "+ guarded DVFS",
          "full": "+ both (ours)", "fcfs_drop": "drop late only", "edf_keep": "deadline-first only",
          "class_priority": "class priority + drop", "naive_dvfs": "unguarded DVFS"}
NAMES = {"brasov": "CityPulse (EdgePlus's Brasov)", "delhi": "AirDelhi (no retraining)"}
# EdgePlus paper (ccppo_edgeplus; A2 = without PSO), from Figs. 4-7, 9-10, 13
PAPER = {
    50000: {"success": 100.0, "lp50": 3.98, "lp95": 10.24, "cloud_pct": float("nan")},
    100000: {"success": 82.5, "lp50": float("nan"), "lp95": 151.0, "cloud_pct": float("nan")},
    150000: {"success": 82.77, "lp50": 53.23, "lp95": 147.09, "cloud_pct": 6.1},
    "150000_A2": {"success": 82.0, "lp50": 54.8, "lp95": 158.9, "cloud_pct": 6.6},
}


def ci95(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float(x.mean()) if len(x) else math.nan), math.nan
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)))


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = n - rank
        prev = min(prev, p[i] * n / k)
        adj[i] = prev
    return adj


def paired_tests(df):
    """Wilcoxon signed-rank of each variant against the baseline, per (dataset, method, load)."""
    rows = []
    for (ds, method, load), g in df.groupby(["eval_dataset", "method", "load"]):
        base = g[g.variant == "baseline"].set_index(["run", "episode"])
        for v in sorted(set(g.variant) - {"baseline"}):
            cur = g[g.variant == v].set_index(["run", "episode"])
            idx = base.index.intersection(cur.index)
            for m in TEST_METRICS:
                d = (cur.loc[idx, m] - base.loc[idx, m]).to_numpy(float)
                d = d[np.isfinite(d)]
                if len(d) < 2 or np.allclose(d, 0):
                    p, r = 1.0, 0.0
                else:
                    p = float(stats.wilcoxon(d, zero_method="zsplit").pvalue)
                    ranks = stats.rankdata(np.abs(d))
                    r = float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())
                rows.append({"eval_dataset": ds, "method": method, "load": load, "variant": v, "metric": m,
                             "n_pairs": len(d), "mean_delta": float(d.mean()) if len(d) else math.nan,
                             "p": p, "rank_biserial": r})
    t = pd.DataFrame(rows)
    if len(t):
        t["p_bh"] = t.groupby("eval_dataset")["p"].transform(lambda s: bh(s.to_numpy()))
    return t


def fmt(m, c, digits=2):
    if not np.isfinite(m):
        return "n/a"
    return f"{m:.{digits}f} ± {c:.{digits}f}" if np.isfinite(c) else f"{m:.{digits}f}"


def summarise(df):
    out = {}
    for key, g in df.groupby(["eval_dataset", "method", "load", "variant"]):
        out["|".join(map(str, key))] = {m: ci95(g[m]) for m in KEY_METRICS if m in g} | {"n": len(g)}
    return out


def table(summary, ds, method, loads, variants, metrics, digits=2):
    head = "| Load | Variant | " + " | ".join(metrics) + " |"
    lines = [head, "|" + "---|" * (len(metrics) + 2)]
    for load in loads:
        for v in variants:
            s = summary.get(f"{ds}|{method}|{load}|{v}")
            if not s:
                continue
            cells = [fmt(*s[m], digits) for m in metrics]
            lines.append(f"| {load // 1000}k | {LABELS.get(v, v)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def figures(df, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    ed = df[df.method == "edgeplus"]
    for ds in sorted(ed.eval_dataset.unique()):
        d = ed[ed.eval_dataset == ds]
        loads = sorted(d.load.unique())
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
        for ax, (m, title) in zip(axes, [("success", "On-time tasks (%)"), ("viol_URLLC", "URLLC missed (%)"),
                                         ("energy_mj", "Energy per task (mJ)")]):
            w = 0.2
            for j, v in enumerate(["baseline", "edf_drop", "guarded_dvfs", "full"]):
                vals = [summary.get(f"{ds}|edgeplus|{L}|{v}", {}).get(m, (math.nan, math.nan)) for L in loads]
                ax.bar(np.arange(len(loads)) + (j - 1.5) * w, [x[0] for x in vals], w,
                       yerr=[0 if not np.isfinite(x[1]) else x[1] for x in vals], label=LABELS[v], capsize=2)
            ax.set_xticks(range(len(loads)), [f"{L // 1000}k" for L in loads])
            ax.set_title(title)
            if m == "success":
                lo = np.nanmin([summary.get(f"{ds}|edgeplus|{L}|baseline", {}).get(m, (100, 0))[0] for L in loads])
                ax.set_ylim(max(0, lo - 10), 100.5)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
        fig.suptitle(f"{NAMES.get(ds, ds)}: EdgePlus with and without the server rules")
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        fig.savefig(FIG / f"main_{ds}.png", dpi=160)
        plt.close(fig)
        # latency by class at the highest load
        L = loads[-1]
        fig, ax = plt.subplots(figsize=(6, 3.4))
        cls = ["lp95_URLLC", "lp95", "lp95_mMTC"]
        for j, v in enumerate(["baseline", "full"]):
            vals = [summary.get(f"{ds}|edgeplus|{L}|{v}", {}).get(m, (math.nan, 0))[0] for m in cls]
            ax.bar(np.arange(3) + (j - 0.5) * 0.35, vals, 0.35, label=LABELS[v])
        ax.set_xticks(range(3), ["URLLC p95", "all tasks p95", "mMTC p95"])
        ax.set_ylabel("latency (s)")
        ax.set_title(f"{NAMES.get(ds, ds).split(' (')[0]}, {L // 1000}k tasks: 95th-percentile latency")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / f"latency_{ds}.png", dpi=160)
        plt.close(fig)
    logs = sorted(glob.glob(str(ROOT / "runs" / "*" / "log.csv")))
    if logs:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        for f in logs:
            lg = pd.read_csv(f)
            name = Path(f).parent.name
            ax.plot(lg["update"], lg["success"], lw=1, label=name)
        ax.set_xlabel("PPO update (one training episode each)")
        ax.set_ylabel("on-time tasks (%)")
        ax.set_title("EdgePlus training curves (stochastic policy)")
        ax.legend(fontsize=6, ncol=3)
        fig.tight_layout()
        fig.savefig(FIG / "training_curves.png", dpi=160)
        plt.close(fig)


def main():
    files = sorted(glob.glob(str(RES / "eval_*.csv")))
    if not files:
        raise SystemExit("no results/eval_*.csv found; run edgeplus.evaluate first")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(["method", "run", "eval_dataset", "load", "variant", "episode"], keep="last")
    df["energy_mj"] = 1000.0 * df["energy_per_task"]
    summary = summarise(df)
    tests = paired_tests(df)
    RES.mkdir(exist_ok=True)
    tests.to_csv(RES / "paired_tests.csv", index=False)
    (RES / "summary.json").write_text(json.dumps(summary, indent=1))
    md = ["# Results", "", "Mean ± 95% CI over all (training seed, test episode) pairs: 3 training seeds x 10 test "
          "episodes = 30 per row. success and viol_* are percentages, latencies (lp*) are seconds, energy_mj is "
          "millijoules per task, cloud_pct and drop_pct are percentages of all tasks.", ""]
    main_m = ["success", "viol_URLLC", "viol_eMBB", "viol_mMTC", "lp50", "lp95", "lp95_URLLC", "lp95_mMTC",
              "energy_mj", "cloud_pct", "drop_pct"]
    for ds in sorted(df.eval_dataset.unique()):
        loads = sorted(df[df.eval_dataset == ds].load.unique())
        md += [f"## {ds}: main comparison (trained EdgePlus policy)", "",
               table(summary, ds, "edgeplus", loads, ["baseline", "edf_drop", "guarded_dvfs", "full"], main_m), ""]
        md += [f"## {ds}: ablations", "",
               table(summary, ds, "edgeplus", loads, list(LABELS), ["success", "viol_URLLC", "viol_eMBB", "viol_mMTC",
                                                                     "lp95", "lp95_URLLC", "energy_mj"]), ""]
        if (df[(df.eval_dataset == ds)].method == "heuristic").any():
            md += [f"## {ds}: routing heuristic (no learning)", "",
                   table(summary, ds, "heuristic", loads, ["baseline", "full"], main_m), ""]
        t = tests[(tests.eval_dataset == ds) & (tests.method == "edgeplus") & (tests.variant == "full")]
        if len(t):
            md += [f"## {ds}: paired tests, full vs baseline (Wilcoxon, BH-corrected)", "",
                   "| Load | Metric | Mean change | p (BH) | rank-biserial r | pairs |", "|---|---|---|---|---|---|"]
            for _, r in t.iterrows():
                md.append(f"| {r.load // 1000}k | {r.metric} | {r.mean_delta:+.3f} | {r.p_bh:.2g} | "
                          f"{r.rank_biserial:+.2f} | {r.n_pairs} |")
            md.append("")
    if "brasov" in set(df.eval_dataset):
        md += ["## Fidelity: our EdgePlus baseline vs the EdgePlus paper (Brasov)", "",
               "| Load | Source | On-time (%) | Lp50 (s) | Lp95 (s) | Cloud (%) |", "|---|---|---|---|---|---|"]
        for L in (50000, 100000, 150000):
            s = summary.get(f"brasov|edgeplus|{L}|baseline")
            if s:
                md.append(f"| {L // 1000}k | ours | {fmt(*s['success'])} | {fmt(*s['lp50'])} | {fmt(*s['lp95'])} | "
                          f"{fmt(*s['cloud_pct'])} |")
            pp = PAPER[L]
            md.append(f"| {L // 1000}k | paper (full EdgePlus) | {pp['success']:.2f} | {pp['lp50']:.2f} | "
                      f"{pp['lp95']:.2f} | {pp['cloud_pct']:.1f} |")
        pp = PAPER["150000_A2"]
        md.append(f"| 150k | paper (EdgePlus without PSO, A2) | {pp['success']:.2f} | {pp['lp50']:.2f} | "
                  f"{pp['lp95']:.2f} | {pp['cloud_pct']:.1f} |")
        md.append("")
    (RES / "tables.md").write_text("\n".join(md).replace("nan", "n/a"))
    figures(df, summary)
    print(f"wrote {RES / 'tables.md'}, {RES / 'summary.json'}, {RES / 'paired_tests.csv'} and figures in {FIG}")


if __name__ == "__main__":
    main()
