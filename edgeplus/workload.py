"""Turn pollution readings into deadline-aware tasks (EdgePlus Algorithm 3) and build the city topology.

Every reading becomes one task. An episode takes N consecutive readings (in time order) and
compresses their real time span into the simulated arrival window (600 s by default), so the
relative burstiness and spatial pattern of the data are kept.
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay

from .config import DATA_DIR

DAY = 86400.0


def _to_km(lon, lat):
    lat0 = np.radians(np.mean(lat))
    return (lon - lon.min()) * 111.32 * np.cos(lat0), (lat - lat.min()) * 110.57


def _scale_into_city(x, y, size_km, lo_q=0.0, hi_q=1.0):
    """Min-max scale coordinates (per axis) into the simulated city, keeping a 0.5 km margin."""
    out = []
    for v in (x, y):
        lo, hi = np.quantile(v, lo_q), np.quantile(v, hi_q)
        out.append(np.clip(0.5 + (v - lo) / max(hi - lo, 1e-9) * (size_km - 1.0), 0.5, size_km - 0.5))
    return out[0].astype(np.float32), out[1].astype(np.float32)


def _moving_mask(dev, t, lon, lat, lag, min_km):
    order = np.lexsort((t, dev))
    lo, la, d = lon[order], lat[order], dev[order]
    dist = np.full(len(order), np.inf)
    dx = (lo[lag:] - lo[:-lag]) * 111.32 * np.cos(np.radians(np.mean(lat)))
    dy = (la[lag:] - la[:-lag]) * 110.57
    same = d[lag:] == d[:-lag]
    dist[lag:] = np.where(same, np.hypot(dx, dy), np.inf)
    keep = np.empty(len(order), bool)
    keep[order] = dist >= min_km
    return keep


def _robust_z(values, train_mask):
    med = np.median(values[train_mask], axis=0)
    q75, q25 = np.percentile(values[train_mask], [75, 25], axis=0)
    return (values - med) / np.maximum((q75 - q25) / 1.349, 1e-6)


class Dataset:
    """Readings sorted by (jittered) time with positions, severity and the train/test boundary."""

    def __init__(self, name, cfg, data_dir=DATA_DIR):
        self.name = name
        w = cfg["workload"]
        size = cfg["city"]["size_km"]
        rng = np.random.default_rng(cfg["city"]["seed"])
        if name == "brasov":
            z = np.load(Path(data_dir) / "brasov.npz")
            site, t, pol = z["site"].astype(np.int64), z["t"].astype(np.float64), z["pol"].astype(np.float32)
            sx, sy = _to_km(z["site_lon"], z["site_lat"])
            sx, sy = _scale_into_city(sx, sy, size)
            boundary = w["brasov_train_days"] * DAY
            train = t < boundary
            # gradient per site (rows are stored site by site in time order)
            grad = np.diff(pol, axis=0, prepend=pol[:1])
            grad[np.r_[True, site[1:] != site[:-1]]] = 0.0
            t_real = t + rng.uniform(0.0, w["brasov_phase_jitter_s"], size=len(sx))[site]
            x, y = sx[site], sy[site]
            level = pol.max(axis=1)
            self.site, self.site_xy = site, np.stack([sx, sy], axis=1)
            self.abs_thresholds = w["brasov_abs_thresholds"]
        elif name == "delhi":
            z = np.load(Path(data_dir) / "delhi.npz")
            dev, t, pol = z["device"].astype(np.int64), z["t"].astype(np.float64), z["pol"].astype(np.float32)
            lon, lat = z["lon"].astype(np.float64), z["lat"].astype(np.float64)
            # Drop readings taken while a bus is parked (moved < 50 m over the previous ~3 minutes).
            # Buses idling at depots would otherwise put ~38% of all readings into three 1 km2 cells.
            moving = _moving_mask(dev, t, lon, lat, lag=60, min_km=0.05)
            self.parked_share = 1.0 - moving.mean()
            dev, t, pol, lon, lat = dev[moving], t[moving], pol[moving], lon[moving], lat[moving]
            x, y = _to_km(lon, lat)
            x, y = _scale_into_city(x, y, size, 0.005, 0.995)
            boundary = 31 * DAY       # Dec 2020 trains, Jan 2021 tests
            train = t < boundary
            order = np.lexsort((t, dev))
            p = pol[order]
            lag = 20                  # 20 samples = about 1 minute per bus
            g = np.zeros_like(p)
            g[lag:] = p[lag:] - p[:-lag]
            same = np.zeros(len(order), bool)
            same[lag:] = dev[order][lag:] == dev[order][:-lag]
            g[~same] = 0.0
            grad = np.empty_like(g)
            grad[order] = g
            t_real = t
            level = pol[:, 0]         # PM2.5 for absolute tiering
            self.site, self.site_xy = None, None
            self.abs_thresholds = w["delhi_abs_thresholds"]
        else:
            raise ValueError(name)
        zs = _robust_z(pol, train)
        gz = _robust_z(grad, train)
        sev = zs.max(axis=1) + 0.5 * np.maximum(gz.max(axis=1), 0.0)
        order = np.argsort(t_real, kind="stable")
        self.t = t_real[order]
        self.x, self.y = x[order], y[order]
        self.sev = sev[order].astype(np.float32)
        self.level = level[order].astype(np.float32)
        self.split_index = int(np.searchsorted(self.t, boundary))
        self._sev_unsorted, self._xy_unsorted = sev, (x, y)
        self.cluster = None

    def ranges(self, split):
        return (0, self.split_index) if split == "train" else (self.split_index, len(self.t))

    def attach(self, topo):
        """Assign every reading to its nearest cluster centroid (the task's home cluster)."""
        c = np.asarray(topo["centroids"], np.float32)
        best = np.zeros(len(self.x), np.int16)
        best_d = np.full(len(self.x), np.inf, np.float32)
        for k in range(len(c)):
            d = (self.x - c[k, 0]) ** 2 + (self.y - c[k, 1]) ** 2
            better = d < best_d
            best[better], best_d[better] = k, d[better]
        self.cluster = best


def build_topology(cfg, ds, path):
    """Weighted K-means (density + exceedance weight, EdgePlus Sec. IV-C) and Delaunay neighbours.

    Brasov clusters fixed sensor sites. Delhi's sensors move, so we cluster a sample of its
    training-period reading positions; density then comes from how often buses pass a place.
    """
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    k = cfg["city"]["clusters"]
    rng = np.random.default_rng(cfg["city"]["seed"])
    if ds.site is not None:
        pts = ds.site_xy.astype(np.float64)
        site_sev = np.bincount(ds.site, weights=ds._sev_unsorted, minlength=len(pts))
        site_sev /= np.maximum(np.bincount(ds.site, minlength=len(pts)), 1)
        w = 1.0 + np.maximum(site_sev, 0.0)
    else:
        idx = rng.choice(ds.split_index, size=min(40000, ds.split_index), replace=False)
        pts = np.stack([ds.x[idx], ds.y[idx]], axis=1).astype(np.float64)
        w = 1.0 + np.maximum(ds.sev[idx].astype(np.float64), 0.0)
    # k-means++ initialisation
    cent = [pts[rng.choice(len(pts), p=w / w.sum())]]
    for _ in range(1, k):
        d2 = np.min([((pts - c) ** 2).sum(1) for c in cent], axis=0) * w
        cent.append(pts[rng.choice(len(pts), p=d2 / d2.sum())])
    cent = np.array(cent)
    for _ in range(cfg["city"]["kmeans_iters"]):
        lab = np.argmin(((pts[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)
        new = np.array([np.average(pts[lab == j], axis=0, weights=w[lab == j]) if np.any(lab == j) else cent[j]
                        for j in range(k)])
        if np.allclose(new, cent):
            break
        cent = new
    nbrs = [set() for _ in range(k)]
    for simplex in Delaunay(cent).simplices:
        for a in simplex:
            for b in simplex:
                if a != b:
                    nbrs[a].add(int(b))
    lab = np.argmin(((pts[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)
    topo = {"dataset": ds.name, "centroids": cent.round(4).tolist(), "neighbors": [sorted(n) for n in nbrs],
            "points_per_cluster": np.bincount(lab, minlength=k).tolist()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(topo, indent=1))
    return topo


def make_episode(ds, cfg, n_tasks, split, index, seed, n_slots=10):
    """Build one episode of `n_tasks` tasks.

    Test episodes are spread evenly over the test period (index 0..n_slots-1, deterministic).
    Training episodes start at a random point of the training period drawn from (seed, index).
    """
    w = cfg["workload"]
    lo, hi = ds.ranges(split)
    span = hi - lo - n_tasks
    if span <= 0:
        raise ValueError(f"{ds.name}/{split} has only {hi - lo} readings, need {n_tasks}")
    if split == "test":
        start = lo + int(round((index % n_slots) * span / max(n_slots - 1, 1)))
    else:
        start = lo + int(np.random.default_rng([seed, index, 7]).integers(0, span))
    sl = slice(start, start + n_tasks)
    rng = np.random.default_rng([seed, index, 11 if split == "test" else 13])
    t = ds.t[sl]
    arrival = (t - t[0]) / max(t[-1] - t[0], 1e-9) * w["window_s"]
    sev = ds.sev[sl].astype(np.float64)
    if w["tiering"] == "percentile":
        order = np.argsort(-(sev + 1e-6 * rng.standard_normal(n_tasks)), kind="stable")
        n_u = int(round(w["class_shares"][0] * n_tasks))
        n_e = int(round(w["class_shares"][1] * n_tasks))
        cls = np.full(n_tasks, 2, np.int64)
        cls[order[:n_u]] = 0
        cls[order[n_u:n_u + n_e]] = 1
    else:
        thr = ds.abs_thresholds
        lvl = ds.level[sl]
        cls = np.where(lvl > thr[0], 0, np.where(lvl > thr[1], 1, 2)).astype(np.int64)
    dl = np.array(w["deadline_s"])[cls]
    deadline = rng.uniform(dl[:, 0], dl[:, 1])
    mean_c = np.array(w["cycles_mean_g"])[cls]
    sig = w["cycles_sigma"]
    cyc = mean_c * np.exp(w["cycles_severity_gain"] * np.clip(sev, -2.0, 3.0)) \
        * rng.lognormal(-0.5 * sig * sig, sig, n_tasks)
    cyc = np.clip(cyc, 0.25 * mean_c, 3.0 * mean_c)
    return {
        "t": arrival, "x": ds.x[sl].astype(np.float64), "y": ds.y[sl].astype(np.float64),
        "cluster": ds.cluster[sl].astype(np.int64), "cls": cls, "deadline": deadline, "cycles": cyc,
        "bits": np.array(w["payload_kb"])[cls] * 8192.0, "mem": np.array(w["memory_mb"])[cls],
        "sev": sev, "span_h": (t[-1] - t[0]) / 3600.0,
    }


def load(cfg, dataset, data_dir=DATA_DIR):
    """Load a dataset with its own cluster layout (same server inventory in every layout)."""
    ds = Dataset(dataset, cfg, data_dir)
    topo = build_topology(cfg, ds, Path(data_dir) / f"topology_{dataset}.json")
    ds.attach(topo)
    return ds, topo
