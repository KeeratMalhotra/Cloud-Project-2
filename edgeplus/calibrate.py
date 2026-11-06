"""Run the routing-order heuristic (no learning) to check the simulator against EdgePlus's numbers.

Targets from EdgePlus (ccppo_edgeplus): 50k -> 100% on time, Lp50 ~4 s, Lp95 ~10 s;
150k -> 82.8% on time, Lp50 ~53 s, Lp95 ~147 s, cloud ~6%.

    python -m edgeplus.calibrate --dataset brasov --episodes 2
"""
import argparse
import time

from .config import load_config, with_rules
from .sim import EdgeSim
from .workload import load, make_episode

COLS = ["success", "viol_URLLC", "viol_eMBB", "viol_mMTC", "lp50", "lp95", "norm_p95", "energy_total",
        "energy_per_task", "cloud_pct", "locality_pct", "drop_pct", "slowed_pct", "fallback_pct"]


def parse_overrides(items):
    """Turn ["servers.stationary_cores=28", ...] into {"servers.stationary_cores": 28, ...}."""
    import tomllib
    out = {}
    for item in items:
        key, val = item.split("=", 1)
        out[key] = tomllib.loads(f"v = {val}")["v"]
    return out


def run_heuristic(cfg, ds, topo, n_tasks, index, variant="baseline"):
    sim = EdgeSim(with_rules(cfg, variant), topo)
    ep = make_episode(ds, cfg, n_tasks, "test", index, 0)
    t0 = time.time()
    pending = sim.reset(ep)
    while pending:
        pending = sim.step(sim.heuristic_action())
    sim.finish_episode()
    out = sim.metrics()
    out["seconds"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="brasov")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--variants", default="baseline")
    ap.add_argument("--loads", default="50000,100000,150000")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. servers.stationary_cores=28")
    args = ap.parse_args()
    cfg = load_config(overrides=parse_overrides(args.set))
    ds, topo = load(cfg, args.dataset)
    print("load    variant        " + " ".join(f"{c:>9s}" for c in COLS) + "   sec")
    for n in [int(x) for x in args.loads.split(",")]:
        for v in args.variants.split(","):
            rows = [run_heuristic(cfg, ds, topo, n, e, v) for e in range(args.episodes)]
            mean = {c: sum(r[c] for r in rows) / len(rows) for c in COLS + ["seconds"]}
            print(f"{n:<7d} {v:14s} " + " ".join(f"{mean[c]:9.2f}" for c in COLS) + f" {mean['seconds']:5.1f}")


if __name__ == "__main__":
    main()
