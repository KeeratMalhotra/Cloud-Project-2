"""Discrete-event simulator of the EdgePlus city with pluggable server-side rules.

Follows EdgePlus Sec. III and IV-C: routing order (stationary in home cluster -> mobile in home
cluster -> mobile in neighbour clusters -> cloud), feasibility masks (deadline, slice budgets,
memory, mobile battery, URLLC reputation quarantine), latency L = tx + move + queue + proc,
energy E = tx + move + compute, ledger-derived reputation and debt, and the Nash / PID / leader
advisory bias added to the policy logits.

Differences from the paper (documented in README): servers are multi-core nodes simulated event by
event (instead of a Kingman queue formula), the mask uses the node's real backlog to estimate the
wait, mobiles relocate between clusters with straight-line paths times a detour factor (instead of
CBS-D* Lite), and the ledger is modelled only through its reputation/debt signals.

Server rules ([rules] in the config):
  queue = fcfs | edf | prio     order in which a node serves waiting tasks
  late  = keep | drop | demote  what to do with a task that can no longer meet its deadline
  dvfs  = off | naive | guarded CPU frequency choice when a task starts
"""
import heapq
import math
from bisect import bisect_right
from collections import deque
from itertools import accumulate

import numpy as np

STAT, MOB, CLOUD = 0, 1, 2
EV_ENQ, EV_DONE, EV_TICK, EV_MOVE = 0, 1, 2, 3
K_MAX = 16          # maximum candidate servers per decision
N_TASK = 13         # task features
N_CAND = 18         # per-candidate features


class Node:
    __slots__ = ("idx", "kind", "home", "cluster", "x", "y", "cores", "ghz", "kappa", "mem_cap",
                 "battery", "busy", "ends", "fifo", "keys", "items", "works", "prefix", "dirty",
                 "pq", "pw", "qwork", "qlen", "demoted", "inflight_work", "inflight_n", "mem_used",
                 "rep", "energy", "moving")

    def __init__(self, idx, kind, cluster, x, y, cores, ghz, kappa, mem_cap, battery):
        self.idx, self.kind, self.home, self.cluster = idx, kind, cluster, cluster
        self.x, self.y, self.cores, self.ghz, self.kappa = x, y, cores, ghz, kappa
        self.mem_cap, self.battery = mem_cap, battery
        self.busy, self.ends = 0, []
        self.fifo = deque()
        self.keys, self.items, self.works, self.prefix, self.dirty = [], [], [], [], False
        self.pq, self.pw = (deque(), deque(), deque()), [0.0, 0.0, 0.0]
        self.qwork, self.qlen, self.demoted = 0.0, 0, deque()
        self.inflight_work, self.inflight_n, self.mem_used = 0.0, 0, 0.0
        self.rep, self.energy, self.moving = 1.0, 0.0, False


class EdgeSim:
    def __init__(self, cfg, topo):
        self.cfg = cfg
        r, s, w, rad, en = cfg["rules"], cfg["servers"], cfg["workload"], cfg["radio"], cfg["energy"]
        self.qmode, self.late, self.dvfs = r["queue"], r["late"], r["dvfs"]
        self.guard, self.slack_use = r["dvfs_guard_s"], r["dvfs_slack_use"]
        self.levels = sorted(en["dvfs_levels"])
        self.cent = [tuple(c) for c in topo["centroids"]]
        self.nbrs = [list(n) for n in topo["neighbors"]]
        self.C = len(self.cent)
        self.window = w["window_s"]
        self.cw = w["class_weight"]
        # radio: per-slice bandwidth, noise, rate(d) = eta_mac * W/Phi * log2(1 + SNR(d))
        self.bw = [b * 1e6 for b in rad["bandwidth_mhz"]]
        self.reuse, self.eta_mac, self.oh = rad["reuse"], rad["eta_mac"], rad["overhead"]
        self.ptx_dbm = rad["tx_power_dbm"] + rad["beamforming_gain_db"]
        self.noise_dbm = [rad["noise_dbm_hz"] + 10 * math.log10(b) + rad["noise_figure_db"] for b in self.bw]
        self.channels = rad["channels"]
        self.range_km, self.speed_kms = s["mobile_range_km"], s["mobile_speed_mps"] / 1000.0
        self.cloud_delay = s["cloud_delay_s"]
        self.txw, self.cloud_j, self.move_j_km = en["tx_power_w"], en["cloud_offload_j"], en["move_j_per_m"] * 1000
        self.dyn, self.stat = 1.0 - en["static_share"], en["static_share"]
        sl = cfg["slices"]
        self.sl_window, self.comp_share, self.radio_share = sl["window_s"], sl["compute_share"], sl["radio_share"]
        lg = cfg["ledger"]
        self.rep_a, self.rep_th, self.quar = lg["rep_alpha"], lg["rep_threshold"], lg["urllc_quarantine"]
        self.debt_t, self.debt_a = lg["debt_target"], lg["debt_alpha"]
        ad = cfg["advisory"]
        self.adv_on = ad["enabled"]
        self.lam = (ad["lambda_nash"], ad["lambda_pid"], ad["lambda_leader"])
        self.kp, self.ki, self.kd = ad["pid_gains"]
        self.setpoint, self.ts = ad["pid_setpoint"], ad["pid_period_s"]
        self.n_alpha, self.n_gamma, self.n_bounds, self.n_share = (ad["nash_alpha"], ad["nash_gamma"],
                                                                   ad["nash_bounds"], ad["nash_share"])
        self.lw = ad["leader_weights"]
        self.redeploy_every, self.detour = ad["redeploy_period_s"], ad["redeploy_detour"]
        self.over_u, self.spare_u = ad["redeploy_overload_u"], ad["redeploy_spare_u"]
        rw = cfg["reward"]
        self.eta, self.e_ref, self.rscale = rw["eta"], rw["energy_ref_j"], rw["scale"]
        self.s = s
        self.kappa = en["kappa_j_per_gcycle"]

    # ------------------------------------------------------------------ setup
    def _build_nodes(self):
        s = self.s
        nodes = []
        self.stat_by_c = [[] for _ in range(self.C)]
        self.mob_in = [[] for _ in range(self.C)]
        for c, (cx, cy) in enumerate(self.cent):
            for j in range(s["stationary_per_cluster"]):
                ang = 2 * math.pi * j / s["stationary_per_cluster"]
                n = Node(len(nodes), STAT, c, cx + 0.3 * math.cos(ang), cy + 0.3 * math.sin(ang),
                         s["stationary_cores"], s["stationary_ghz"][j % len(s["stationary_ghz"])],
                         self.kappa[0], s["stationary_mem_gb"] * 1024, math.inf)
                nodes.append(n)
                self.stat_by_c[c].append(n)
            for _ in range(s["mobile_per_cluster"]):
                n = Node(len(nodes), MOB, c, cx, cy, s["mobile_cores"], s["mobile_ghz"], self.kappa[1],
                         s["mobile_mem_gb"] * 1024, s["mobile_battery_j"])
                nodes.append(n)
                self.mob_in[c].append(n)
        self.cloud = Node(len(nodes), CLOUD, -1, 0.0, 0.0, s["cloud_cores"], s["cloud_ghz"], self.kappa[2],
                          s["cloud_mem_gb"] * 1024, math.inf)
        nodes.append(self.cloud)
        self.nodes = nodes
        self.cores_in = [sum(n.cores for n in self.stat_by_c[c]) + sum(m.cores for m in self.mob_in[c])
                         for c in range(self.C)]

    def reset(self, ep):
        """Start an episode. Returns True if a decision is pending (see the o_* attributes)."""
        self.n = n = len(ep["t"])
        self.T = ep["t"].tolist()
        self.X, self.Y = ep["x"].tolist(), ep["y"].tolist()
        self.CL, self.K = ep["cluster"].tolist(), ep["cls"].tolist()
        self.D, self.CY = ep["deadline"].tolist(), ep["cycles"].tolist()
        self.B, self.M = ep["bits"].tolist(), ep["mem"].tolist()
        self.SEV = ep["sev"].tolist()
        self.DA = [t + d for t, d in zip(self.T, self.D)]
        self.node_of = [-1] * n
        self.tier_of = [-1] * n
        self.proc = [0.0] * n
        self.finish = [math.nan] * n
        self.end_of = [0.0] * n
        self.dropped = [False] * n
        self.ontime = [False] * n
        self.energy = [0.0] * n
        self.freq = [1.0] * n
        self.rew = np.zeros(n, np.float32)
        self.fallback_n = 0
        self.reloc_energy = 0.0
        self._build_nodes()
        C = self.C
        self.load = [0.0] * C
        self.pid_i, self.pid_e = [0.0] * C, [0.0] * C
        self.u = [0.0] * C
        self.share = [1.0] * C
        self.log_share = [0.0] * C
        self.lead = [self.stat_by_c[c][0].idx for c in range(C)]
        self.theta, self.debt = [1.0] * C, [0.0] * C
        self.sl_q = [deque() for _ in range(C)]
        self.sl_comp = [[0.0, 0.0, 0.0] for _ in range(C)]
        self.sl_air = [[0.0, 0.0, 0.0] for _ in range(C)]
        self._update_budgets()
        self.heap = [(self.ts, 0, EV_TICK, 0, 0)]
        self.seq = 1
        self.next_redeploy = self.redeploy_every
        self.now = 0.0
        self.ptr = 0
        self.active = 0
        self.o_task = np.zeros(N_TASK, np.float32)
        self.o_cand = np.zeros((K_MAX, N_CAND), np.float32)
        self.o_mask = np.zeros(K_MAX, bool)
        self.o_bias = np.zeros(K_MAX, np.float32)
        self.o_glob = np.zeros(7 * C + 2 + N_TASK, np.float32)
        return self._advance()

    def _update_budgets(self):
        self.comp_budget = [[sh * self.sl_window * self.cores_in[c] for sh in self.comp_share] for c in range(self.C)]
        self.air_budget = [[sh * self.sl_window * self.channels for sh in self.radio_share] for c in range(self.C)]

    # ------------------------------------------------------------------ radio and queues
    def _rate(self, k, d_km):
        pl = 128.1 + 37.6 * math.log10(max(d_km, 0.05))
        snr = 10 ** ((self.ptx_dbm - pl - self.noise_dbm[k]) / 10)
        return self.eta_mac * self.bw[k] / self.reuse * math.log2(1 + snr)

    def _work_ahead(self, node, dabs, k):
        if self.qmode == "fcfs":
            return node.qwork
        if self.qmode == "edf":
            if not node.keys:
                return 0.0
            if node.dirty:
                node.prefix = list(accumulate(node.works))
                node.dirty = False
            p = bisect_right(node.keys, dabs)
            return node.prefix[p - 1] if p else 0.0
        pw = node.pw
        return pw[0] if k == 0 else (pw[0] + pw[1] if k == 1 else pw[0] + pw[1] + pw[2])

    def _est_wait(self, node, dabs, k):
        free = node.cores - node.busy
        if node.qlen == 0 and node.inflight_n < free:
            return 0.0
        ahead = self._work_ahead(node, dabs, k) + node.inflight_work
        minres = (min(node.ends) - self.now) if free <= 0 else 0.0
        return max(0.0, minres) + ahead / node.cores

    def _push(self, node, i):
        w = self.proc[i]
        if self.qmode == "fcfs":
            node.fifo.append(i)
        elif self.qmode == "edf":
            d = self.DA[i]
            p = bisect_right(node.keys, d)
            node.keys.insert(p, d)
            node.items.insert(p, i)
            node.works.insert(p, w)
            node.dirty = True
        else:
            k = self.K[i]
            node.pq[k].append(i)
            node.pw[k] += w
        node.qwork += w
        node.qlen += 1

    def _pop(self, node):
        """Next task to serve on this node, applying the late-task rule."""
        now, late = self.now, self.late
        while node.qlen:
            if self.qmode == "fcfs":
                i = node.fifo.popleft()
            elif self.qmode == "edf":
                node.keys.pop(0)
                i = node.items.pop(0)
                node.works.pop(0)
                node.dirty = True
            else:
                k = 0 if node.pq[0] else (1 if node.pq[1] else 2)
                i = node.pq[k].popleft()
                node.pw[k] -= self.proc[i]
            node.qwork -= self.proc[i]
            node.qlen -= 1
            if late != "keep" and now + self.proc[i] > self.DA[i]:
                if late == "drop":
                    self._drop(node, i)
                    continue
                node.demoted.append(i)
                continue
            return i
        if node.demoted:
            return node.demoted.popleft()
        return None

    def _choose_freq(self, node, i, proc):
        slack = self.DA[i] - self.now - proc
        if slack <= 0:
            return 1.0
        if self.dvfs == "guarded":
            if node.qlen or node.inflight_n:
                return 1.0
            budget = min(self.slack_use * slack, self.guard)
        else:
            budget = self.slack_use * slack
        for f in self.levels:
            if proc / f - proc <= budget:
                return f
        return 1.0

    def _start(self, node, i):
        proc = self.proc[i]
        if self.late == "drop" and self.now + proc > self.DA[i]:
            self._drop(node, i)
            return False
        f = self._choose_freq(node, i, proc) if self.dvfs != "off" else 1.0
        end = self.now + proc / f
        node.busy += 1
        node.ends.append(end)
        e = self.CY[i] * node.kappa * (self.dyn * f * f + self.stat / f)
        self.energy[i] += e
        if node.kind == MOB:
            node.energy += e
        self.freq[i] = f
        self.end_of[i] = end
        heapq.heappush(self.heap, (end, self.seq, EV_DONE, node.idx, i))
        self.seq += 1
        return True

    def _pull(self, node):
        while node.busy < node.cores:
            j = self._pop(node)
            if j is None:
                return
            self._start(node, j)

    def _outcome(self, node, i, on, lat_norm):
        k = self.K[i]
        w = self.cw[k]
        eta = self.eta
        r = (eta[0] * w if on else -eta[1] * w) - eta[2] * lat_norm - eta[3] * self.energy[i] / self.e_ref
        self.rew[i] += self.rscale * r
        self.ontime[i] = on
        node.rep += self.rep_a * ((1.0 if on else 0.0) - node.rep)
        c = self.CL[i]
        self.theta[c] += self.debt_a * ((1.0 if on else 0.0) - self.theta[c])
        self.debt[c] = max(0.0, self.debt_t - self.theta[c])
        node.mem_used -= self.M[i]
        self.active -= 1

    def _drop(self, node, i):
        self.dropped[i] = True
        self._outcome(node, i, False, 2.0)

    # ------------------------------------------------------------------ controllers (advisory layer)
    def _tick(self):
        C = self.C
        for c in range(C):
            members = self.stat_by_c[c] + self.mob_in[c]
            cores = sum(m.cores for m in members)
            self.cores_in[c] = cores
            self.load[c] = min(2.0, sum(m.busy + m.qlen + m.inflight_n for m in members) / max(cores, 1))
            # PID on utilisation headroom (positive u = spare capacity)
            e = self.setpoint - self.load[c]
            self.pid_i[c] = max(-1.0 / self.ki, min(1.0 / self.ki, self.pid_i[c] + e * self.ts))
            de = (e - self.pid_e[c]) / self.ts
            self.pid_e[c] = e
            self.u[c] = max(-1.0, min(1.0, self.kp * e + self.ki * self.pid_i[c] + self.kd * de))
            # leader: spare capacity and remaining energy
            best, best_s = members[0].idx, -1e9
            for m in members:
                ratio = min(1.0, (m.busy + m.qlen) / m.cores)
                eratio = 1.0 if m.kind == STAT else max(0.0, 1.0 - m.energy / m.battery)
                sc = self.lw[0] * (1.0 - ratio) + self.lw[1] * eratio
                if sc > best_s:
                    best, best_s = m.idx, sc
            self.lead[c] = best
        self._nash()
        # slice windows
        horizon = self.now - self.sl_window
        for c in range(C):
            q = self.sl_q[c]
            while q and q[0][0] < horizon:
                _, k, comp, air = q.popleft()
                self.sl_comp[c][k] -= comp
                self.sl_air[c][k] -= air
        self._update_budgets()
        if self.now >= self.next_redeploy:
            self.next_redeploy += self.redeploy_every
            self._redeploy()

    def _nash(self):
        lo, hi = self.n_bounds
        s = self.share
        base = self.load
        for _ in range(20):
            eff = list(base)
            for c in range(self.C):
                excess = max(0.0, base[c] - 1.0)
                if excess > 0:
                    tot = sum(s[n] for n in self.nbrs[c]) or 1.0
                    for n in self.nbrs[c]:
                        eff[n] += self.n_share * excess * s[n] / tot
            new = [min(hi, max(lo, 1.0 - self.n_alpha * eff[c] / (2 * self.n_gamma))) for c in range(self.C)]
            delta = max(abs(a - b) for a, b in zip(new, s))
            s = new
            if delta < 1e-3:
                break
        self.share = s
        self.log_share = [math.log(v) for v in s]

    def _redeploy(self):
        order = sorted(range(self.C), key=lambda c: self.u[c])
        for c in order:
            if self.u[c] >= self.over_u:
                break
            best, best_u = None, self.spare_u
            for nb in self.nbrs[c]:
                if self.u[nb] <= best_u:
                    continue
                for m in self.mob_in[nb]:
                    if not m.moving and m.busy == 0 and m.qlen == 0 and m.inflight_n == 0 and m.energy < m.battery:
                        best, best_u = m, self.u[nb]
                        break
            if best is None:
                continue
            cx, cy = self.cent[c]
            dist = math.hypot(best.x - cx, best.y - cy) * self.detour
            best.moving = True
            self.mob_in[best.cluster].remove(best)
            e = self.move_j_km * dist
            best.energy += e
            self.reloc_energy += e
            heapq.heappush(self.heap, (self.now + dist / self.speed_kms, self.seq, EV_MOVE, best.idx, c))
            self.seq += 1

    # ------------------------------------------------------------------ event loop
    def _process(self, ev):
        t, _, typ, a, b = ev
        self.now = t
        if typ == EV_DONE:
            node = self.nodes[a]
            node.busy -= 1
            node.ends.remove(self.end_of[b])
            self.finish[b] = t
            lat = t - self.T[b]
            self._outcome(node, b, lat <= self.D[b] + 1e-9, min(lat / self.D[b], 2.0))
            self._pull(node)
        elif typ == EV_ENQ:
            node = self.nodes[a]
            node.inflight_work -= self.proc[b]
            node.inflight_n -= 1
            if node.busy < node.cores and node.qlen == 0:
                if not self._start(node, b):
                    self._pull(node)
            else:
                self._push(node, b)
                self._pull(node)
        elif typ == EV_TICK:
            self._tick()
            if self.ptr < self.n or self.active > 0:
                heapq.heappush(self.heap, (t + self.ts, self.seq, EV_TICK, 0, 0))
                self.seq += 1
        else:  # EV_MOVE
            m = self.nodes[a]
            m.moving, m.cluster = False, b
            m.x, m.y = self.cent[b][0] + 0.2, self.cent[b][1] + 0.2
            self.mob_in[b].append(m)

    def _advance(self):
        heap = self.heap
        while True:
            nxt = self.T[self.ptr] if self.ptr < self.n else math.inf
            if heap and heap[0][0] <= nxt:
                self._process(heapq.heappop(heap))
                continue
            if self.ptr >= self.n:
                return False
            self.now = nxt
            self._observe(self.ptr)
            return True

    def finish_episode(self):
        """Run until every dispatched task has finished or been dropped."""
        heap = self.heap
        while heap and (self.active > 0 or heap[0][2] != EV_TICK):
            self._process(heapq.heappop(heap))

    # ------------------------------------------------------------------ observation and dispatch
    def _observe(self, i):
        c, k = self.CL[i], self.K[i]
        tx_, ty_ = self.X[i], self.Y[i]
        dl, dabs, cyc, bits, mem = self.D[i], self.DA[i], self.CY[i], self.B[i] * self.oh, self.M[i]
        cx, cy = self.cent[c]
        d_home = math.hypot(tx_ - cx, ty_ - cy)
        rate_home = self._rate(k, d_home)
        cands = list(self.stat_by_c[c])
        tiers = [0] * len(cands)
        for m in self.mob_in[c]:
            if not m.moving:
                cands.append(m)
                tiers.append(1)
        for nb in self.nbrs[c]:
            for m in self.mob_in[nb]:
                if not m.moving:
                    cands.append(m)
                    tiers.append(2)
        cands = cands[:K_MAX - 1]
        tiers = tiers[:K_MAX - 1]
        cands.append(self.cloud)
        tiers.append(3)
        feats, valid, lats, res_ok_l, bias = [], [], [], [], []
        info = []
        rep_q = k == 0
        lam_n, lam_p, lam_l = self.lam
        for node, tier in zip(cands, tiers):
            if node.kind == STAT:
                txr = bits / rate_home
                move_km = 0.0
                delay = txr
            elif node.kind == MOB:
                dm = math.hypot(tx_ - node.x, ty_ - node.y)
                if dm <= self.range_km:
                    txr = bits / self._rate(k, dm)
                    move_km = 0.0
                else:
                    txr = bits / self._rate(k, self.range_km)
                    move_km = dm - self.range_km
                delay = txr + move_km / self.speed_kms
            else:
                txr = bits / rate_home
                move_km = 0.0
                delay = txr + self.cloud_delay
            proc = cyc / node.ghz
            wait = max(0.0, self._est_wait(node, dabs, k) - delay)
            lat = delay + wait + proc
            e_est = cyc * node.kappa + self.txw * txr + (self.cloud_j if node.kind == CLOUD else 0.0) \
                + self.move_j_km * move_km
            cn = node.cluster
            res_ok = node.mem_used + mem <= node.mem_cap and (node.kind != MOB or node.energy + e_est <= node.battery)
            if node.kind == CLOUD:
                slice_ok, debt, u, ls, lead = True, 0.0, 0.0, 0.0, 0.0
            else:
                slice_ok = (self.sl_comp[cn][k] + proc <= self.comp_budget[cn][k]
                            and self.sl_air[cn][k] + txr <= self.air_budget[cn][k])
                debt, u, ls = self.debt[cn], self.u[cn], self.log_share[cn]
                lead = 1.0 if self.lead[cn] == node.idx else 0.0
            ok = res_ok and slice_ok and lat <= dl and not (rep_q and node.rep < self.quar)
            valid.append(ok)
            res_ok_l.append(res_ok)
            lats.append(lat)
            bias.append((lam_n * ls + lam_p * u + lam_l * lead) if self.adv_on else 0.0)
            info.append((node, tier, delay, txr, move_km, proc, e_est))
            feats.append((tier == 0, tier == 1, tier == 2, tier == 3, wait / 30.0, lat / dl,
                          max(-2.0, (dl - lat) / dl), proc / 10.0, delay / 10.0, node.busy / node.cores,
                          min(10.0, (node.qlen + node.inflight_n) / node.cores), node.rep, debt, u, ls, lead,
                          e_est / self.e_ref,
                          1.0 if node.kind != MOB else max(0.0, 1.0 - node.energy / node.battery)))
        nc = len(cands)
        if not any(valid):
            self.fallback_n += 1
            pool = [j for j in range(nc) if res_ok_l[j]] or list(range(nc))
            best = min(pool, key=lambda j: lats[j])
            valid = [j == best for j in range(nc)]
        self._info = info
        self._lats = lats
        self._tiers = tiers
        oc, om, ob = self.o_cand, self.o_mask, self.o_bias
        oc[:] = 0.0
        om[:] = False
        ob[:] = 0.0
        oc[:nc] = feats
        om[:nc] = valid
        ob[:nc] = bias
        self.o_n = nc
        nvalid = sum(valid)
        cap = self.comp_budget[c][k]
        task = (k == 0, k == 1, k == 2, dl / 600.0, cyc / 10.0, self.SEV[i] / 3.0, self.load[c], self.u[c],
                self.log_share[c], self.debt[c], self.sl_comp[c][k] / cap if cap > 0 else 1.0, nvalid / K_MAX,
                self.now / self.window)
        self.o_task[:] = task
        g = self.o_glob
        C = self.C
        for j in range(C):
            b = 7 * j
            cb = self.comp_budget[j]
            g[b:b + 7] = (self.load[j], self.u[j], self.log_share[j], self.debt[j],
                          self.sl_comp[j][0] / cb[0], self.sl_comp[j][1] / cb[1], self.sl_comp[j][2] / cb[2])
        g[7 * C] = (self.cloud.busy + self.cloud.qlen) / self.cloud.cores
        g[7 * C + 1] = self.now / self.window
        g[7 * C + 2:] = self.o_task

    def step(self, a):
        """Dispatch the pending task to candidate `a`, then advance to the next decision."""
        i = self.ptr
        node, tier, delay, txr, move_km, proc, e_est = self._info[a]
        self.node_of[i], self.tier_of[i], self.proc[i] = node.idx, tier, proc
        node.inflight_work += proc
        node.inflight_n += 1
        node.mem_used += self.M[i]
        extra = self.txw * txr + (self.cloud_j if node.kind == CLOUD else 0.0) + self.move_j_km * move_km
        self.energy[i] += extra
        if node.kind == MOB:
            node.energy += extra
        k = self.K[i]
        if node.kind != CLOUD:
            cn = node.cluster
            self.sl_comp[cn][k] += proc
            self.sl_air[cn][k] += txr
            self.sl_q[cn].append((self.now, k, proc, txr))
        eta = self.eta
        r = -eta[4] * (node.kind == CLOUD) - eta[5] * max(0.0, self.rep_th - node.rep) \
            - eta[6] * (self.debt[node.cluster] if node.kind != CLOUD else 0.0)
        self.rew[i] = self.rscale * r
        heapq.heappush(self.heap, (self.now + delay, self.seq, EV_ENQ, node.idx, i))
        self.seq += 1
        self.active += 1
        self.ptr += 1
        return self._advance()

    # ------------------------------------------------------------------ helpers
    def heuristic_action(self):
        """Routing-order first fit: lowest tier with a valid server, earliest estimated finish within it."""
        best, key = 0, None
        for j in range(self.o_n):
            if self.o_mask[j]:
                kj = (self._tiers[j], self._lats[j])
                if key is None or kj < key:
                    best, key = j, kj
        return best

    def metrics(self):
        n = self.n
        on = np.array(self.ontime)
        drop = np.array(self.dropped)
        cls = np.array(self.K)
        fin = np.array(self.finish)
        t = np.array(self.T)
        d = np.array(self.D)
        tier = np.array(self.tier_of)
        lat = fin - t
        done = ~drop
        out = {"tasks": n, "success": 100.0 * on.mean(), "drop_pct": 100.0 * drop.mean(),
               "throughput": on.sum() / self.window}
        ld = lat[done]
        out["lp50"] = float(np.percentile(ld, 50)) if ld.size else math.nan
        out["lp95"] = float(np.percentile(ld, 95)) if ld.size else math.nan
        norm = np.where(done, lat / d, 99.0)   # dropped tasks count as 99x their deadline
        out["norm_p95"] = float(np.percentile(norm, 95))
        for k, name in enumerate(self.cfg["workload"]["class_names"]):
            m = cls == k
            out[f"viol_{name}"] = 100.0 * (1.0 - on[m].mean()) if m.any() else math.nan
            lk = lat[m & done]
            out[f"lp95_{name}"] = float(np.percentile(lk, 95)) if lk.size else math.nan
        e = np.array(self.energy)
        total = e.sum() + self.reloc_energy
        out["energy_total"] = total
        out["energy_per_task"] = total / n
        out["energy_per_ontime"] = total / max(1, on.sum())
        out["cloud_pct"] = 100.0 * (tier == 3).mean()
        out["locality_pct"] = 100.0 * np.isin(tier, (0, 1)).mean()
        rates = [on[np.array(self.CL) == c].mean() for c in range(self.C) if (np.array(self.CL) == c).any()]
        rates = np.array(rates)
        out["jain"] = float(rates.sum() ** 2 / (len(rates) * (rates ** 2).sum())) if rates.size else math.nan
        f = np.array(self.freq)
        out["slowed_pct"] = 100.0 * (f < 1.0).mean()
        out["fallback_pct"] = 100.0 * self.fallback_n / n
        return out
