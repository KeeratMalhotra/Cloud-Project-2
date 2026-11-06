"""Extra explanatory figures for the paper (topologies, workload, worked examples, detailed results).

    python -m edgeplus.figures

Needs data/ (python -m edgeplus.data), trained runs in runs/ and results/summary.json
(python -m edgeplus.analyze). Writes PNG files to paper/figures/ and prints the numbers quoted in
the paper text.
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from .config import ROOT, load_config, with_rules  # noqa: E402
from .policy import NumpyPolicy, load_policy  # noqa: E402
from .sim import EdgeSim  # noqa: E402
from .workload import load, make_episode  # noqa: E402

OUT = ROOT / "paper" / "figures"
CLS = ["URLLC", "eMBB", "mMTC"]
CLS_COLOR = {"URLLC": "#d62728", "eMBB": "#1f77b4", "mMTC": "#2ca02c"}
BASE_C, OURS_C = "#1f77b4", "#d62728"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
                     "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight"})


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", OUT / name)


# --------------------------------------------------------------------------- diagrams
def system_overview():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 62)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec="#333333", fs=9, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", wrap=True)

    def arrow(x1, y1, x2, y2, color="#333333", style="-|>", lw=1.4):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14, color=color, lw=lw))

    blue, orange, grey = "#dbe9f6", "#fde3cf", "#f2f2f2"
    box(1, 38, 19, 17, "Air-quality sensors\n(fixed sites or\nsensors on buses)\n\nevery reading\nbecomes one task", grey, fs=8.5)
    for i in range(6):
        ax.plot(4.5 + i * 2.4, 39.6, "o", ms=3.5, color="#555555")
    box(25, 33, 27, 24, "EdgePlus scheduler\n(in the task's home cluster)\n\n1. The feasibility mask removes\n"
        "servers that cannot finish\nthe task in time.\n2. A learned policy (PPO)\nscores the remaining servers.\n"
        "3. An advisory bias (Nash, PID,\nleader) nudges the choice.", blue, fs=8.2)
    arrow(20.5, 46, 24.5, 46)
    ax.text(22.5, 47.5, "task", ha="center", fontsize=8)
    opts = [("Stationary edge servers\nin the home cluster", 51), ("Mobile edge server\nin the home cluster", 42.5),
            ("Mobile servers in\nneighbouring clusters", 34), ("Cloud data centre\n(+2 s extra delay)", 25.5)]
    for text, y in opts:
        box(59, y - 3.2, 23, 6.8, text, blue, fs=8.5)
        arrow(52.5, 45, 58.5, y + 0.2, color="#555555", lw=1.0)
    ax.text(70.5, 58.5, "routing order: tried from top to bottom", ha="center", fontsize=8, style="italic")
    # zoom into one server
    ax.add_patch(Rectangle((52, 1), 57, 18, fc="#fffaf3", ec="#e6550d", lw=1.6, ls="--"))
    ax.text(80.5, 17, "Inside every server (this is where our two rules work)", ha="center", fontsize=9,
            fontweight="bold", color="#a63603")
    ax.plot([83.2, 96, 96], [25.5, 25.5, 19.2], color="#e6550d", lw=1.4, ls="--")
    for j in range(5):
        ax.add_patch(Rectangle((54.5 + j * 3.4, 6.5), 3.0, 5.5,
                               fc=["#d62728", "#1f77b4", "#2ca02c", "#1f77b4", "#2ca02c"][j], ec="white", alpha=0.85))
    ax.text(63, 4.4, "waiting queue", ha="center", fontsize=8)
    arrow(72, 9.2, 75, 9.2)
    for j in range(3):
        ax.add_patch(Rectangle((75.5, 4 + j * 3.4), 8, 2.8, fc="#cccccc", ec="#555555"))
        ax.text(79.5, 5.4 + j * 3.4, "CPU core", ha="center", va="center", fontsize=7.5)
    box(85.2, 9.6, 23.0, 5.4, "Rule 1: serve the earliest deadline\nfirst; drop tasks that can no\nlonger finish in time", orange, fs=7.4)
    box(85.2, 2.6, 23.0, 5.4, "Rule 2: slow the CPU down only\nwhen nothing else is waiting\n(guarded DVFS)", orange, fs=7.4)
    ax.add_patch(Rectangle((1, 1), 2.5, 2.5, fc=blue, ec="#333333"))
    ax.text(4.5, 2.2, "part of EdgePlus", va="center", fontsize=8)
    ax.add_patch(Rectangle((22, 1), 2.5, 2.5, fc=orange, ec="#333333"))
    ax.text(25.5, 2.2, "added in this work", va="center", fontsize=8)
    ax.text(1, 28, "Every task has a class and a deadline:\n"
            "  URLLC (urgent alerts):     5-15 s\n  eMBB (routine analysis):  30-120 s\n  mMTC (logging):          120-600 s",
            fontsize=8.5, va="top", family="monospace")
    _save(fig, "system_overview.png")


def toy_example():
    """Five tasks on a one-core server under three queueing rules (worked example in the paper)."""
    tasks = {  # name: (class, arrival, work, relative deadline)
        "A": ("mMTC", 0.0, 4.0, 300.0), "B": ("eMBB", 0.5, 3.0, 60.0), "C": ("URLLC", 1.0, 2.0, 8.0),
        "D": ("URLLC", 1.5, 2.0, 6.0), "E": ("URLLC", 2.0, 3.0, 3.0)}

    def run(order, drop):
        t, done, waiting, sched = 0.0, set(), [], []
        names = sorted(tasks, key=lambda n: tasks[n][1])
        while len(done) < len(tasks):
            waiting = [n for n in names if n not in done and tasks[n][1] <= t]
            if not waiting:
                t = min(tasks[n][1] for n in names if n not in done)
                continue
            key = (lambda n: tasks[n][1]) if order == "fcfs" else (lambda n: tasks[n][1] + tasks[n][3])
            n = min(waiting, key=key)
            cls, arr, work, dl = tasks[n]
            if drop and t + work > arr + dl:
                sched.append((n, t, t, "dropped"))
                done.add(n)
                continue
            sched.append((n, t, t + work, "on time" if t + work <= arr + dl else "late"))
            done.add(n)
            t += work
        return sched

    rows = [("First come, first served (EdgePlus)", run("fcfs", False)),
            ("Earliest deadline first, no dropping", run("edf", False)),
            ("Earliest deadline first + drop (ours)", run("edf", True))]
    fig, axes = plt.subplots(3, 1, figsize=(9, 5.4), sharex=True)
    for ax, (title, sched) in zip(axes, rows):
        ok = sum(1 for s in sched if s[3] == "on time")
        for n, s, e, status in sched:
            cls = tasks[n][0]
            if status == "dropped":
                end = max(e for _, _, e, st in sched if st != "dropped")
                ax.text(end + 0.25, 0.5, f"{n} dropped at {s:.0f} s: it could not\nfinish by its {tasks[n][1] + tasks[n][3]:.0f} s deadline",
                        va="center", fontsize=8, color="#555555", style="italic")
                continue
            ax.add_patch(Rectangle((s, 0.15), e - s, 0.7, fc=CLS_COLOR[cls], ec="black",
                                   hatch="//" if status == "late" else None, alpha=0.9))
            ax.text((s + e) / 2, 0.5, f"{n}\n{'late' if status == 'late' else 'ok'}", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_title(f"{title}: {ok} of 5 tasks on time", loc="left", fontsize=10)
    axes[-1].set_xlim(0, 15)
    axes[-1].set_xlabel("time (s)")
    handles = [Rectangle((0, 0), 1, 1, fc=CLS_COLOR[c]) for c in CLS] + [Rectangle((0, 0), 1, 1, fc="white", ec="k", hatch="//")]
    fig.legend(handles, CLS + ["missed its deadline"], loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, "toy_queue.png")
    for title, sched in rows:
        print(title, [(n, round(s, 1), round(e, 1), st) for n, s, e, st in sched])


def dvfs_curve(cfg):
    a = 1.0 - cfg["energy"]["static_share"]
    phi = np.linspace(0.3, 1.0, 200)
    energy = a * phi ** 2 + (1 - a) / phi
    levels = np.array(cfg["energy"]["dvfs_levels"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    ax.plot(phi, energy, color="#333333", label="energy of one task")
    ax.plot(phi, a * phi ** 2, ls="--", color="#1f77b4", label="dynamic part (falls with speed)")
    ax.plot(phi, (1 - a) / phi, ls=":", color="#ff7f0e", label="static part (grows with run time)")
    ax.scatter(levels, a * levels ** 2 + (1 - a) / levels, color="#d62728", zorder=3, label="speeds we allow")
    ax.set_xlabel("CPU speed (fraction of full speed)")
    ax.set_ylabel("energy (fraction of full-speed energy)")
    ax.set_title("Slower is cheaper, down to about 55% speed")
    ax.set_ylim(0, 1.42)
    ax.legend(fontsize=7.5, loc="upper left", ncol=2)
    ax = axes[1]
    work, slack, guard = 3.0, 6.0, cfg["rules"]["dvfs_guard_s"]
    budget = min(cfg["rules"]["dvfs_slack_use"] * slack, guard)
    f = min(l for l in levels if work / l - work <= budget)
    ax.add_patch(Rectangle((0, 1.2), work, 0.6, fc="#999999", ec="k"))
    ax.text(work / 2, 1.5, f"full speed: {work:.1f} s", ha="center", va="center", fontsize=8.5, color="white")
    ax.add_patch(Rectangle((0, 0.3), work / f, 0.6, fc="#2ca02c", ec="k"))
    ax.text(work / f / 2, 0.6, f"at {f:.0%} speed: {work / f:.1f} s", ha="center", va="center", fontsize=8.5, color="white")
    ax.axvline(work + slack, color="#d62728", ls="--")
    ax.text(work + slack - 0.12, 2.05, "deadline", color="#d62728", ha="right", fontsize=8.5)
    ax.annotate("", xy=(work + guard, 0.15), xytext=(work, 0.15), arrowprops=dict(arrowstyle="<->", color="#e6550d"))
    ax.text(work + guard / 2, 0.02, f"max extra {guard:.1f} s", ha="center", fontsize=8, color="#e6550d")
    ax.set_xlim(0, work + slack + 0.8)
    ax.set_ylim(0, 2.3)
    ax.set_yticks([])
    ax.set_xlabel("time after the task starts (s)")
    e_full = 1.0
    e_slow = a * f ** 2 + (1 - a) / f
    ax.set_title(f"Example: same task, {100 * (1 - e_slow / e_full):.0f}% less energy")
    fig.tight_layout()
    _save(fig, "dvfs_energy.png")
    for l in levels:
        print(f"speed {l:.1f}: time x{1 / l:.2f}, energy x{a * l * l + (1 - a) / l:.3f}")


# --------------------------------------------------------------------------- data figures
def topology(cfg):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2))
    for ax, name, title in zip(axes, ["brasov", "delhi"], ["CityPulse (EdgePlus's Brasov data)", "AirDelhi (sensors on buses)"]):
        ds, topo = load(cfg, name)
        cent = np.array(topo["centroids"])
        rng = np.random.default_rng(0)
        if name == "brasov":
            xy, cnt = np.unique(ds.site_xy.round(4), axis=0, return_counts=True)
            lab = np.argmin(((xy[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)
            ax.scatter(xy[:, 0], xy[:, 1], s=8 + 6 * cnt, c=lab, cmap="tab10", alpha=0.75, edgecolors="none")
        else:
            idx = rng.choice(len(ds.x), 25000, replace=False)
            ax.scatter(ds.x[idx], ds.y[idx], s=1.5, c=ds.cluster[idx], cmap="tab10", alpha=0.4, edgecolors="none")
        for a, nb in enumerate(topo["neighbors"]):
            for b in nb:
                if b > a:
                    ax.plot(cent[[a, b], 0], cent[[a, b], 1], color="#777777", lw=0.8, ls="--", zorder=1)
        for c, (cx, cy) in enumerate(cent):
            for j in range(4):
                ang = 2 * math.pi * j / 4
                ax.plot(cx + 0.3 * math.cos(ang), cy + 0.3 * math.sin(ang), "s", ms=4, color="#1f3b73", zorder=3)
            ax.plot(cx, cy, "^", ms=7, color="#2ca02c", mec="k", mew=0.5, zorder=4)
            ax.text(cx + 0.35, cy + 0.35, str(c), fontsize=9, fontweight="bold", zorder=5)
        share = np.bincount(ds.cluster, minlength=len(cent)) / len(ds.cluster)
        ax.set_title(f"{title}\nbusiest cluster has {share.max():.0%} of tasks, quietest {share.min():.1%}", fontsize=10)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")
        ax.set_xlabel("km")
        ax.set_ylabel("km")
        print(name, "task share per cluster:", np.round(share * 100, 1).tolist())
    handles = [plt.Line2D([], [], marker="s", ls="", color="#1f3b73", label="stationary edge server (4 per cluster)"),
               plt.Line2D([], [], marker="^", ls="", color="#2ca02c", mec="k", label="mobile edge server (starts at the centre)"),
               plt.Line2D([], [], ls="--", color="#777777", label="neighbouring clusters (can share mobile servers)"),
               plt.Line2D([], [], marker="o", ls="", color="#888888", label="sensor readings, coloured by cluster")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    _save(fig, "topology.png")


def workload(cfg):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for name, label, color in [("brasov", "CityPulse", "#1f77b4"), ("delhi", "AirDelhi", "#d62728")]:
        ds, topo = load(cfg, name)
        ep = make_episode(ds, cfg, 150000, "test", 0, 0)
        cnt, edges = np.histogram(ep["t"], bins=60, range=(0, 600))
        axes[0].plot(edges[:-1] + 5, cnt / 10.0, color=color, label=f"{label} (episode covers {ep['span_h']:.0f} h of real data)")
        share = np.bincount(ds.cluster, minlength=10) / len(ds.cluster)
        off = -0.2 if name == "brasov" else 0.2
        axes[1].bar(np.arange(10) + off, share * 100, 0.4, color=color, label=label)
        print(name, "150k test episode 0: arrivals per second min/mean/max",
              round(cnt.min() / 10, 1), round(cnt.mean() / 10, 1), round(cnt.max() / 10, 1))
    axes[0].set_xlabel("simulated time (s)")
    axes[0].set_ylabel("new tasks per second")
    axes[0].set_title("How tasks arrive (150,000-task episode)")
    axes[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False)
    axes[0].set_ylim(0, None)
    axes[1].axhline(10, color="#555555", ls=":", lw=1)
    axes[1].text(7.0, 10.5, "even split (10%)", ha="center", fontsize=8)
    axes[1].set_xticks(range(10))
    axes[1].set_xlabel("cluster")
    axes[1].set_ylabel("share of all tasks (%)")
    axes[1].set_title("Where tasks come from")
    axes[1].legend()
    fig.tight_layout()
    _save(fig, "workload.png")


def _detailed(cfg, dataset, run, n_tasks, variant, episode=0):
    ds, topo = load(cfg, dataset)
    model, norms, meta, _ = load_policy(ROOT / "runs" / run / "policy.pt")
    pol = NumpyPolicy(model, norms)
    sim = EdgeSim(with_rules(cfg, variant), topo)
    ep = make_episode(ds, cfg, n_tasks, "test", episode, 0)
    pending = sim.reset(ep)
    while pending:
        pending = sim.step(pol.greedy(sim))
    sim.finish_episode()
    return {"t": np.array(sim.T), "lat": np.array(sim.finish) - np.array(sim.T), "dl": np.array(sim.D),
            "cls": np.array(sim.K), "on": np.array(sim.ontime), "drop": np.array(sim.dropped),
            "cluster": np.array(sim.CL), "freq": np.array(sim.freq), "tier": np.array(sim.tier_of),
            "metrics": sim.metrics()}


def detailed_results(cfg):
    runs = {"cp150": ("brasov", "brasov_150k_s0", 150000), "dl150": ("delhi", "brasov_150k_s0", 150000),
            "cp50": ("brasov", "brasov_50k_s0", 50000)}
    data = {k: {v: _detailed(cfg, *spec, v) for v in ("baseline", "full")} for k, spec in runs.items()}
    # latency CDF per class, CityPulse 150k
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    for ax, k in zip(axes, range(3)):
        for v, color, label in [("baseline", BASE_C, "EdgePlus"), ("full", OURS_C, "ours")]:
            d = data["cp150"][v]
            m = (d["cls"] == k) & ~d["drop"]
            lat = np.sort(d["lat"][m])
            ax.plot(lat, np.arange(1, len(lat) + 1) / len(lat) * 100, color=color, label=label)
        lo, hi = cfg["workload"]["deadline_s"][k]
        ax.axvspan(lo, hi, color="#cccccc", alpha=0.4, label="deadline range")
        ax.set_xscale("log")
        ax.set_title(f"{CLS[k]} tasks")
        ax.set_xlabel("latency (s, log scale)")
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("% of tasks finished within this time")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    _save(fig, "latency_cdf.png")
    # on-time rate over the episode
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), sharey=True)
    for ax, key, title in zip(axes, ["cp150", "dl150"], ["CityPulse, 150,000 tasks", "AirDelhi, 150,000 tasks"]):
        for v, color, label in [("baseline", BASE_C, "EdgePlus"), ("full", OURS_C, "ours")]:
            d = data[key][v]
            bins = np.arange(0, 601, 20)
            idx = np.clip(np.digitize(d["t"], bins) - 1, 0, len(bins) - 2)
            rate = np.array([d["on"][idx == b].mean() * 100 if (idx == b).any() else np.nan for b in range(len(bins) - 1)])
            ax.plot(bins[:-1] + 10, rate, color=color, marker="o", ms=3, label=label)
        ax.set_title(title)
        ax.set_xlabel("arrival time of the task (s)")
        ax.set_ylim(0, 102)
    axes[0].set_ylabel("tasks finished on time (%)")
    axes[0].legend()
    fig.tight_layout()
    _save(fig, "over_time.png")
    # per-cluster on-time
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), sharey=True)
    for ax, key, title in zip(axes, ["cp150", "dl150"], ["CityPulse, 150,000 tasks", "AirDelhi, 150,000 tasks"]):
        for j, (v, color, label) in enumerate([("baseline", BASE_C, "EdgePlus"), ("full", OURS_C, "ours")]):
            d = data[key][v]
            rates = [d["on"][d["cluster"] == c].mean() * 100 for c in range(10)]
            ax.bar(np.arange(10) + (j - 0.5) * 0.4, rates, 0.4, color=color, label=label)
            print(key, v, "on-time per cluster:", np.round(rates, 1).tolist())
        share = np.bincount(data[key]["baseline"]["cluster"], minlength=10) / len(data[key]["baseline"]["cluster"])
        for c in range(10):
            ax.text(c, 2, f"{share[c]:.0%}", ha="center", fontsize=7, color="white", fontweight="bold")
        ax.set_xticks(range(10))
        ax.set_xlabel("cluster (white number = its share of tasks)")
        ax.set_title(title)
    axes[0].set_ylabel("tasks finished on time (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, "per_cluster.png")
    # chosen CPU speeds
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    levels = cfg["energy"]["dvfs_levels"]
    for j, (key, label, color) in enumerate([("cp50", "50,000 tasks", "#2ca02c"), ("cp150", "150,000 tasks", "#9467bd")]):
        f = data[key]["full"]["freq"]
        shares = [(np.isclose(f, l)).mean() * 100 for l in levels]
        ax.bar(np.arange(len(levels)) + (j - 0.5) * 0.4, shares, 0.4, color=color, label=label)
        print(key, "speed shares (%):", dict(zip(levels, np.round(shares, 1))))
    ax.set_xticks(range(len(levels)), [f"{l:.0%}" for l in levels])
    ax.set_xlabel("CPU speed used for a task")
    ax.set_ylabel("share of tasks (%)")
    ax.set_title("Which speeds the guarded rule picks (CityPulse)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "speeds.png")
    for key, dd in data.items():
        for v, d in dd.items():
            m = d["metrics"]
            print(key, v, f"on-time {m['success']:.2f}% median latency {np.nanmedian(d['lat'][~d['drop']]):.1f}s",
                  {CLS[k]: round(float(np.median(d["lat"][(d["cls"] == k) & ~d["drop"]])), 1) for k in range(3)})


def summary_figures():
    s = json.loads((ROOT / "results" / "summary.json").read_text())
    g = lambda ds, L, v, m: s[f"{ds}|edgeplus|{L}|{v}"][m][0]
    # ablation
    variants = [("baseline", "EdgePlus (as published)"), ("fcfs_drop", "drop late tasks only"), ("edf_keep", "deadline first only"),
                ("class_priority", "fixed class priority + drop"), ("edf_drop", "deadline first + drop"),
                ("naive_dvfs", "unguarded slow-down"), ("guarded_dvfs", "guarded slow-down"), ("full", "both rules (ours)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    vals = [g("brasov", 150000, v, "success") for v, _ in variants]
    colors = ["#1f77b4"] + ["#bbbbbb"] * 6 + ["#d62728"]
    axes[0].bar(range(len(variants)), vals, color=colors)
    for i, v in enumerate(vals):
        axes[0].text(i, v + 1, f"{v:.1f}", ha="center", fontsize=8)
    axes[0].set_xticks(range(len(variants)), [l for _, l in variants], fontsize=8, rotation=35, ha="right")
    axes[0].set_ylabel("tasks on time (%)")
    axes[0].set_ylim(0, 108)
    axes[0].set_title("On-time tasks for each rule set (CityPulse, 150,000 tasks)")
    w = 0.27
    for j, k in enumerate(CLS):
        axes[1].bar(np.arange(len(variants)) + (j - 1) * w, [g("brasov", 150000, v, f"viol_{k}") for v, _ in variants], w,
                    color=CLS_COLOR[k], label=k)
    axes[1].set_xticks(range(len(variants)), [l for _, l in variants], fontsize=8, rotation=35, ha="right")
    axes[1].set_ylabel("missed deadlines (%)")
    axes[1].set_title("Who misses the deadline")
    axes[1].legend()
    fig.tight_layout()
    _save(fig, "ablation.png")
    # energy savings vs load
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
    loads = [50000, 100000, 150000]
    for ds, label, color in [("brasov", "CityPulse", "#1f77b4"), ("delhi", "AirDelhi", "#d62728")]:
        for v, ls, vlabel in [("guarded_dvfs", "--", "slow-down only"), ("full", "-", "both rules")]:
            sav = [100 * (1 - g(ds, L, v, "energy_mj") / g(ds, L, "baseline", "energy_mj")) for L in loads]
            axes[0].plot([L // 1000 for L in loads], sav, ls=ls, marker="o", color=color, label=f"{label}, {vlabel}")
        axes[1].plot([L // 1000 for L in loads], [g(ds, L, "full", "slowed_pct") for L in loads], marker="o",
                     color=color, label=label)
    axes[0].set_xlabel("load (thousand tasks per episode)")
    axes[0].set_ylabel("energy saved per task (%)")
    axes[0].set_title("Energy saving shrinks as servers get busier")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, None)
    axes[1].set_xlabel("load (thousand tasks per episode)")
    axes[1].set_ylabel("tasks run below full speed (%)")
    axes[1].set_title("How often the CPU is slowed")
    axes[1].legend()
    axes[1].set_ylim(0, 100)
    for ax in axes:
        ax.set_xticks([50, 100, 150])
    fig.tight_layout()
    _save(fig, "energy_savings.png")
    # per-class p95 latency for both datasets
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
    for ax, ds, title in zip(axes, ["brasov", "delhi"], ["CityPulse, 150,000 tasks", "AirDelhi, 150,000 tasks"]):
        keys = ["lp95_URLLC", "lp95_eMBB", "lp95_mMTC", "lp95"]
        names = ["URLLC", "eMBB", "mMTC", "all tasks"]
        for j, (v, color, label) in enumerate([("baseline", BASE_C, "EdgePlus"), ("full", OURS_C, "ours")]):
            vals = [g(ds, 150000, v, k) for k in keys]
            ax.bar(np.arange(4) + (j - 0.5) * 0.38, vals, 0.38, color=color, label=label)
            for i, val in enumerate(vals):
                ax.text(i + (j - 0.5) * 0.38, val + 8, f"{val:.0f}", ha="center", fontsize=7.5)
        ax.set_xticks(range(4), names)
        ax.set_ylabel("95th-percentile latency (s)")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    _save(fig, "latency_by_class.png")
    # training curves: mean and range per load
    import pandas as pd
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    for L, color in [(50, "#2ca02c"), (100, "#1f77b4"), (150, "#d62728")]:
        logs = [pd.read_csv(f) for f in sorted((ROOT / "runs").glob(f"brasov_{L}k_s*/log.csv"))]
        arr = np.array([lg["success"].to_numpy() for lg in logs])
        u = logs[0]["update"].to_numpy()
        ax.plot(u, arr.mean(0), color=color, label=f"{L},000 tasks")
        ax.fill_between(u, arr.min(0), arr.max(0), color=color, alpha=0.2)
    ax.set_xlabel("training round (one simulated episode each)")
    ax.set_ylabel("tasks on time during training (%)")
    ax.set_title("Training the EdgePlus model (3 seeds, shaded = range)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "training.png")


def main():
    cfg = load_config()
    system_overview()
    toy_example()
    dvfs_curve(cfg)
    topology(cfg)
    workload(cfg)
    summary_figures()
    detailed_results(cfg)


if __name__ == "__main__":
    main()
