"""Evaluate frozen policies (and the routing heuristic) under each server-rule variant.

    python -m edgeplus.evaluate --dataset brasov --runs runs/brasov_*k_s* --episodes 10
    python -m edgeplus.evaluate --dataset delhi  --runs runs/brasov_*k_s* --episodes 10   # zero-shot

Every policy is run greedily on the same test episodes with each variant (the policy weights never
change between variants, so differences come only from the server rules). Results are appended
per episode to a CSV for analyze.py.
"""
import argparse
import csv
import glob
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .config import ROOT, VARIANTS, load_config, with_rules
from .policy import NumpyPolicy, load_policy
from .sim import EdgeSim
from .workload import load, make_episode

_G = {}


def _init(dataset, overrides):
    cfg = load_config(overrides=overrides)
    ds, topo = load(cfg, dataset)
    _G.update(cfg=cfg, ds=ds, topo=topo, dataset=dataset, policies={})


def _policy(run):
    if run not in _G["policies"]:
        model, norms, meta, _ = load_policy(Path(run) / "policy.pt")
        _G["policies"][run] = (NumpyPolicy(model, norms), meta)
    return _G["policies"][run]


def _job(args):
    method, run, n_tasks, variant, episode = args
    cfg = with_rules(_G["cfg"], variant)
    sim = EdgeSim(cfg, _G["topo"])
    ep = make_episode(_G["ds"], cfg, n_tasks, "test", episode, 0)
    t0 = time.time()
    pending = sim.reset(ep)
    if method == "heuristic":
        while pending:
            pending = sim.step(sim.heuristic_action())
        train_seed, train_set = -1, ""
    else:
        pol, meta = _policy(run)
        while pending:
            pending = sim.step(pol.greedy(sim))
        train_seed, train_set = meta["seed"], meta["dataset"]
    sim.finish_episode()
    elapsed = time.time() - t0
    m = sim.metrics()
    m.update(method=method, run=Path(run).name if run else "", eval_dataset=_G["dataset"], train_dataset=train_set,
             load=n_tasks, train_seed=train_seed, variant=variant, episode=episode,
             decision_ms=1000.0 * elapsed / n_tasks)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="brasov", choices=["brasov", "delhi"])
    ap.add_argument("--runs", nargs="*", default=[], help="run directories (globs allowed)")
    ap.add_argument("--variants", default="all")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--heuristic-loads", default="", help="also evaluate the routing heuristic at these loads")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    variants = list(VARIANTS) if args.variants == "all" else args.variants.split(",")
    runs = sorted({r for pat in args.runs for r in glob.glob(pat) if (Path(r) / "policy.pt").exists()})
    jobs = []
    for run in runs:
        meta = json.loads((Path(run) / "config.json").read_text())["meta"]
        jobs += [("edgeplus", run, meta["load"], v, e) for v in variants for e in range(args.episodes)]
    for n in [int(x) for x in args.heuristic_loads.split(",") if x]:
        jobs += [("heuristic", "", n, v, e) for v in variants for e in range(args.episodes)]
    if not jobs:
        raise SystemExit("nothing to evaluate (no runs with policy.pt and no --heuristic-loads)")
    out = Path(args.out or ROOT / "results" / f"eval_{args.dataset}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():   # resume: skip episodes already in the file
        with open(out) as f:
            for r in csv.DictReader(f):
                done.add((r["method"], r["run"], int(r["load"]), r["variant"], int(r["episode"])))
    jobs = [j for j in jobs if (j[0], Path(j[1]).name if j[1] else "", j[2], j[3], j[4]) not in done]
    # longest episodes first so the pool stays busy
    jobs.sort(key=lambda j: -j[2])
    print(f"{len(jobs)} episodes to simulate ({len(done)} already done) with {args.workers} workers -> {out}")
    if not jobs:
        return
    t0 = time.time()
    keys = ["method", "run", "eval_dataset", "train_dataset", "load", "train_seed", "variant", "episode"]
    f = open(out, "a", newline="")
    writer = None
    with Pool(args.workers, initializer=_init, initargs=(args.dataset, {})) as pool:
        for i, row in enumerate(pool.imap_unordered(_job, jobs, chunksize=1)):
            if writer is None:
                fields = keys + [k for k in row if k not in keys]
                writer = csv.DictWriter(f, fieldnames=fields)
                if f.tell() == 0:
                    writer.writeheader()
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
            f.flush()
            if (i + 1) % max(1, len(jobs) // 20) == 0:
                print(f"  {i + 1}/{len(jobs)} done ({time.time() - t0:.0f}s)", flush=True)
    f.close()
    print(f"wrote {len(jobs)} rows in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
