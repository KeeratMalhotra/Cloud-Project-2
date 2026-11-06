"""Train the EdgePlus policy (masked cooperative PPO with centralized critic) in the simulator.

    python -m edgeplus.train --dataset brasov --load 150000 --seed 0

Each update simulates one full training episode (one decision per task), fills in each decision's
reward once its task has finished, and runs clipped PPO with GAE (EdgePlus Eq. 8). Training uses
the baseline server rules unless --variant is given. Checkpoints are written every few updates and
training resumes from the last checkpoint if the run directory already exists.
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .calibrate import parse_overrides
from .config import RUNS_DIR, load_config, with_rules
from .policy import ActorCritic, NumpyPolicy, RunningNorm, load_policy, save_policy
from .sim import K_MAX, N_CAND, N_TASK, EdgeSim
from .workload import load, make_episode


class Buffer:
    def __init__(self, n, glob_dim):
        self.task = np.zeros((n, N_TASK), np.float32)
        self.cand = np.zeros((n, K_MAX, N_CAND), np.float32)
        self.mask = np.zeros((n, K_MAX), bool)
        self.bias = np.zeros((n, K_MAX), np.float32)
        self.glob = np.zeros((n, glob_dim), np.float32)
        self.ncand = np.zeros(n, np.int16)
        self.act = np.zeros(n, np.int64)
        self.logp = np.zeros(n, np.float32)
        self.val = np.zeros(n, np.float32)


def rollout(sim, pol, ep, rng, buf):
    """Simulate one episode with the stochastic policy, storing raw observations."""
    pending = sim.reset(ep)
    t = 0
    while pending:
        a, logp, val, _, _, _ = pol.act(sim, rng)
        buf.task[t] = sim.o_task
        buf.cand[t] = sim.o_cand
        buf.mask[t] = sim.o_mask
        buf.bias[t] = sim.o_bias
        buf.glob[t] = sim.o_glob
        buf.ncand[t] = sim.o_n
        buf.act[t], buf.logp[t], buf.val[t] = a, logp, val
        pending = sim.step(a)
        t += 1
    sim.finish_episode()
    return t, sim.rew[:t].copy()


def gae(rew, val, gamma, lam):
    adv = np.zeros_like(rew)
    last = 0.0
    nxt = 0.0
    for t in range(len(rew) - 1, -1, -1):
        delta = rew[t] + gamma * nxt - val[t]
        last = delta + gamma * lam * last
        adv[t] = last
        nxt = val[t]
    return adv, adv + val


def ppo_update(model, opt, buf, n, norms, adv, ret, p, rng):
    nt, nc, ng = norms
    task = torch.from_numpy(nt(buf.task[:n]))
    cand = torch.from_numpy(nc(buf.cand[:n]))
    exist = np.arange(K_MAX)[None, :] < buf.ncand[:n, None]
    cand[torch.from_numpy(~exist)] = 0.0
    glob = torch.from_numpy(ng(buf.glob[:n]))
    mask = torch.from_numpy(buf.mask[:n])
    bias = torch.from_numpy(buf.bias[:n])
    act = torch.from_numpy(buf.act[:n])
    old = torch.from_numpy(buf.logp[:n])
    a = torch.from_numpy(((adv - adv.mean()) / (adv.std() + 1e-8)).astype(np.float32))
    r = torch.from_numpy(ret.astype(np.float32))
    choice = torch.from_numpy(buf.mask[:n].sum(1) > 1)       # forced decisions carry no policy gradient
    stats = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "kl": 0.0, "clipfrac": 0.0, "steps": 0}
    for _ in range(p["epochs"]):
        perm = torch.from_numpy(rng.permutation(n))
        kls = []
        for s in range(0, n, p["minibatch"]):
            idx = perm[s:s + p["minibatch"]]
            lg = model.logits(task[idx], cand[idx], bias[idx], mask[idx])
            lp_all = torch.log_softmax(lg, dim=-1)
            lp = lp_all.gather(1, act[idx, None]).squeeze(1)
            ratio = torch.exp(lp - old[idx])
            w = choice[idx].float()
            denom = w.sum().clamp(min=1.0)
            surr = torch.min(ratio * a[idx], ratio.clamp(1 - p["clip"], 1 + p["clip"]) * a[idx])
            pg = -(surr * w).sum() / denom
            prob = lp_all.exp()
            ent = (-(prob * lp_all.clamp(min=-30)).sum(-1) * w).sum() / denom
            kl = ((old[idx] - lp) * w).sum() / denom
            vf = ((model.value(glob[idx]) - r[idx]) ** 2).mean()
            loss = pg + p["value_coef"] * vf - p["entropy"] * ent + p["kl_coef"] * kl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), p["max_grad_norm"])
            opt.step()
            kls.append(kl.item())
            stats["pg"] += pg.item()
            stats["vf"] += vf.item()
            stats["ent"] += ent.item()
            stats["clipfrac"] += (((ratio - 1).abs() > p["clip"]).float() * w).sum().item() / denom.item()
            stats["steps"] += 1
        stats["kl"] = float(np.mean(kls))
        if stats["kl"] > 1.5 * p["target_kl"]:
            break
    for k in ("pg", "vf", "ent", "clipfrac"):
        stats[k] /= max(stats["steps"], 1)
    return stats


def update_norms(norms, buf, n):
    nt, nc, ng = norms
    nt.update(buf.task[:n])
    exist = np.arange(K_MAX)[None, :] < buf.ncand[:n, None]
    nc.update(buf.cand[:n][exist])
    ng.update(buf.glob[:n])


def train(args):
    cfg = load_config(overrides=parse_overrides(args.set))
    if args.variant:
        cfg = with_rules(cfg, args.variant)
    p = cfg["ppo"]
    updates = args.updates or p["updates"]
    torch.set_num_threads(args.threads)
    out = Path(args.out or RUNS_DIR / f"{args.dataset}_{args.load // 1000}k_s{args.seed}"
               + (f"_{args.variant}" if args.variant else ""))
    out.mkdir(parents=True, exist_ok=True)
    ds, topo = load(cfg, args.dataset)
    sim = EdgeSim(cfg, topo)
    glob_dim = 7 * len(topo["centroids"]) + 2 + N_TASK
    torch.manual_seed(args.seed)
    model = ActorCritic(glob_dim, p["actor_hidden"], p["critic_hidden"])
    opt = torch.optim.Adam(model.parameters(), lr=p["lr"], eps=1e-5)
    norms = [RunningNorm(N_TASK), RunningNorm(N_CAND), RunningNorm(glob_dim)]
    start = 0
    ck_path = out / "policy.pt"
    if (out / "DONE").exists():
        print(f"{out} already finished")
        return out
    if ck_path.exists():
        model, norms, meta, ck = load_policy(ck_path)
        opt = torch.optim.Adam(model.parameters(), lr=p["lr"], eps=1e-5)
        opt.load_state_dict(ck["opt"])
        start = meta["update"]
        print(f"resuming {out} from update {start}")
    meta = {"dataset": args.dataset, "load": args.load, "seed": args.seed, "variant": args.variant or "baseline",
            "glob_dim": glob_dim, "actor_hidden": p["actor_hidden"], "critic_hidden": p["critic_hidden"],
            "rules": cfg["rules"], "updates": updates}
    (out / "config.json").write_text(json.dumps({"cfg": cfg, "meta": meta}, indent=1, default=str))
    buf = Buffer(args.load, glob_dim)
    rng = np.random.default_rng([args.seed, start])
    log_path = out / "log.csv"
    cols = ["update", "success", "viol_URLLC", "viol_eMBB", "viol_mMTC", "cloud_pct", "energy_per_task",
            "mean_reward", "pg", "vf", "ent", "kl", "clipfrac", "rollout_s", "update_s"]
    kept = []
    if start > 0 and log_path.exists():   # drop log rows written after the last checkpoint
        with open(log_path) as f:
            kept = [r for r in csv.reader(f)][1:]
        kept = [r for r in kept if r and int(r[0]) <= start]
    logf = open(log_path, "w", newline="")
    writer = csv.writer(logf)
    writer.writerow(cols)
    writer.writerows(kept)
    for u in range(start, updates):
        t0 = time.time()
        if u == 0 and args.warmup_norms:
            # fit the observation normaliser on one heuristic episode before the first update
            ep = make_episode(ds, cfg, args.load, "train", 10_000, args.seed)
            pending, t = sim.reset(ep), 0
            while pending:
                buf.task[t], buf.cand[t], buf.glob[t], buf.ncand[t] = sim.o_task, sim.o_cand, sim.o_glob, sim.o_n
                pending = sim.step(sim.heuristic_action())
                t += 1
            update_norms(norms, buf, t)
        pol = NumpyPolicy(model, norms)
        ep = make_episode(ds, cfg, args.load, "train", u, args.seed)
        n, rew = rollout(sim, pol, ep, rng, buf)
        m = sim.metrics()
        t1 = time.time()
        adv, ret = gae(rew, buf.val[:n], p["gamma"], p["gae_lambda"])
        st = ppo_update(model, opt, buf, n, norms, adv, ret, p, rng)
        update_norms(norms, buf, n)
        t2 = time.time()
        row = [u + 1, m["success"], m["viol_URLLC"], m["viol_eMBB"], m["viol_mMTC"], m["cloud_pct"],
               m["energy_per_task"], float(rew.mean()), st["pg"], st["vf"], st["ent"], st["kl"], st["clipfrac"],
               t1 - t0, t2 - t1]
        writer.writerow([f"{x:.5g}" if isinstance(x, float) else x for x in row])
        logf.flush()
        print(f"[{out.name}] update {u + 1}/{updates} success {m['success']:.2f}% URLLC viol {m['viol_URLLC']:.1f}% "
              f"cloud {m['cloud_pct']:.1f}% reward {rew.mean():.3f} ent {st['ent']:.3f} kl {st['kl']:.4f} "
              f"({t1 - t0:.0f}s + {t2 - t1:.0f}s)", flush=True)
        if (u + 1) % p["checkpoint_every"] == 0 or u + 1 == updates:
            meta["update"] = u + 1
            tmp = out / "policy.pt.tmp"
            save_policy(tmp, model, norms, meta)
            ck = torch.load(tmp, weights_only=False)
            ck["opt"] = opt.state_dict()
            torch.save(ck, tmp)
            os.replace(tmp, ck_path)
    logf.close()
    (out / "DONE").write_text("finished\n")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="brasov", choices=["brasov", "delhi"])
    ap.add_argument("--load", type=int, default=150000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=0, help="number of PPO updates (default from config)")
    ap.add_argument("--variant", default="", help="train with these server rules (default: baseline)")
    ap.add_argument("--out", default="")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--no-warmup-norms", dest="warmup_norms", action="store_false")
    ap.add_argument("--set", action="append", default=[], help="config override, e.g. ppo.lr=1e-4")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
