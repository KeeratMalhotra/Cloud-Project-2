"""EdgePlus policy: masked cooperative PPO actor with a centralized critic (EdgePlus Sec. III-E, IV-E).

The actor scores every candidate server from (task features, candidate features); parameters are
shared by all cluster actors, which see their own cluster's state in the task features. The
advisory bias (Nash / PID / leader) is added to the logits and infeasible servers are masked out
before the softmax (EdgePlus Eq. 9-10). The critic sees the global city state.

Rollouts run with a NumPy copy of the weights (fast single-decision inference); updates use PyTorch.
"""
import numpy as np
import torch
import torch.nn as nn

from .sim import K_MAX, N_CAND, N_TASK

NEG = -1e9


class RunningNorm:
    """Running mean/variance (training statistics only; frozen at evaluation)."""

    def __init__(self, dim):
        self.mean = np.zeros(dim, np.float64)
        self.var = np.ones(dim, np.float64)
        self.count = 1e-4

    def update(self, x):
        x = np.asarray(x, np.float64).reshape(-1, self.mean.size)
        if len(x) == 0:
            return
        bm, bv, bc = x.mean(0), x.var(0), len(x)
        delta, tot = bm - self.mean, self.count + bc
        self.mean = self.mean + delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x):
        return ((x - self.mean) / np.sqrt(self.var + 1e-8)).clip(-10, 10).astype(np.float32)

    def state(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load(self, s):
        self.mean, self.var, self.count = np.asarray(s["mean"]), np.asarray(s["var"]), float(s["count"])


def _mlp(i, h, o, out_gain):
    net = nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))
    for j, m in enumerate(net):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, out_gain if j == len(net) - 1 else np.sqrt(2))
            nn.init.zeros_(m.bias)
    return net


class ActorCritic(nn.Module):
    def __init__(self, glob_dim, actor_hidden=64, critic_hidden=128):
        super().__init__()
        self.actor = _mlp(N_TASK + N_CAND, actor_hidden, 1, 0.01)
        self.critic = _mlp(glob_dim, critic_hidden, 1, 1.0)

    def logits(self, task, cand, bias, mask):
        b, k, _ = cand.shape
        x = torch.cat([task.unsqueeze(1).expand(b, k, task.shape[-1]), cand], dim=-1)
        z = self.actor(x).squeeze(-1) + bias
        return z.masked_fill(~mask, NEG)

    def value(self, glob):
        return self.critic(glob).squeeze(-1)


class NumpyPolicy:
    """Fast inference copy of the actor and critic for rollouts and evaluation."""

    def __init__(self, model, norms):
        self.w = [(m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy())
                  for m in model.actor if isinstance(m, nn.Linear)]
        self.v = [(m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy())
                  for m in model.critic if isinstance(m, nn.Linear)]
        self.nt, self.nc, self.ng = norms
        self._x = np.zeros((K_MAX, N_TASK + N_CAND), np.float32)

    def act(self, sim, rng=None, greedy=False):
        """Return (action, logp, value, task_n, cand_n, glob_n) for the pending decision."""
        k = sim.o_n
        tn = self.nt(sim.o_task)
        cn = self.nc(sim.o_cand)
        x = self._x
        x[:, :N_TASK] = tn
        x[:, N_TASK:] = cn
        h = x[:k]
        (w1, b1), (w2, b2), (w3, b3) = self.w
        z = (np.tanh(np.tanh(h @ w1 + b1) @ w2 + b2) @ w3 + b3)[:, 0] + sim.o_bias[:k]
        mask = sim.o_mask[:k]
        z = np.where(mask, z, NEG)
        z = z - z.max()
        p = np.exp(z)
        p /= p.sum()
        if greedy:
            a = int(np.argmax(p))
        else:
            a = int(min(np.searchsorted(np.cumsum(p), rng.random()), k - 1))
            if not mask[a]:
                a = int(np.argmax(p))
        logp = float(np.log(max(p[a], 1e-12)))
        gn = self.ng(sim.o_glob)
        (v1, c1), (v2, c2), (v3, c3) = self.v
        val = float((np.tanh(np.tanh(gn @ v1 + c1) @ v2 + c2) @ v3 + c3)[0])
        return a, logp, val, tn, cn, gn


def save_policy(path, model, norms, meta):
    torch.save({"model": model.state_dict(), "norms": [n.state() for n in norms], "meta": meta}, path)


def load_policy(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    model = ActorCritic(meta["glob_dim"], meta["actor_hidden"], meta["critic_hidden"])
    model.load_state_dict(ck["model"])
    norms = [RunningNorm(N_TASK), RunningNorm(N_CAND), RunningNorm(meta["glob_dim"])]
    for n, s in zip(norms, ck["norms"]):
        n.load(s)
    return model, norms, meta, ck
