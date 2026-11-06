"""Download the two pollution datasets and convert them into compact NumPy caches.

Run:  python -m edgeplus.data --dataset all

Brasov (CityPulse): the dataset EdgePlus calls "Brasov" (449 sites, Aug-Oct 2014). Its CityPulse
download is the file named citypulse_pollution_csv_data_aarhus_aug_oct_2014.tar.gz, and CityPulse
describes the values as generated (a bounded random walk updated every 5 minutes).

AirDelhi (IIT Delhi, NeurIPS 2023, CC BY 4.0): PM sensors on public buses, Nov 2020 - Jan 2021.
We use the raw PM files for the months listed in the config and clean them ourselves.
"""
import argparse
import io
import json
import tarfile
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, load_config

CITYPULSE_URL = ("http://iot.ee.surrey.ac.uk:8080/datasets/pollution/"
                 "citypulse_pollution_csv_data_aarhus_aug_oct_2014.tar.gz")
HF_TREE_URL = "https://huggingface.co/api/datasets/sachin-iitd/DelhiPollDataset/tree/main?recursive=true"
HF_FILE_URL = "https://huggingface.co/datasets/sachin-iitd/DelhiPollDataset/resolve/main/"

BRASOV_POLLUTANTS = ["ozone", "particullate_matter", "carbon_monoxide", "sulfure_dioxide", "nitrogen_dioxide"]
DELHI_POLLUTANTS = ["pm2_5", "pm10"]
DELHI_BOX = {"lat": (28.40, 28.90), "lon": (76.85, 77.40)}   # Delhi NCT; drops bad GPS fixes


def _download(url, dest, timeout=900):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(dest)
    return dest


def prepare_brasov(data_dir=DATA_DIR):
    """Build data/brasov.npz: one row per reading (site, time, five AQI values) plus site coordinates."""
    data_dir = Path(data_dir)
    out = data_dir / "brasov.npz"
    if out.exists():
        print(f"[brasov] cache exists: {out}")
        return out
    archive = _download(CITYPULSE_URL, data_dir / "raw" / "citypulse_pollution_csv.tar.gz")
    sites, lon, lat, t_all, p_all, s_all = [], [], [], [], [], []
    t0 = pd.Timestamp("2014-08-01 00:00:00")
    with tarfile.open(archive, "r:gz") as tar:
        members = sorted((m for m in tar.getmembers() if m.name.endswith(".csv")), key=lambda m: m.name)
        for sid, m in enumerate(members):
            df = pd.read_csv(io.BytesIO(tar.extractfile(m).read()))
            ts = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")
            t_all.append(((ts - t0).dt.total_seconds()).to_numpy(np.int32))
            p_all.append(df[BRASOV_POLLUTANTS].to_numpy(np.uint8))
            s_all.append(np.full(len(df), sid, np.int16))
            sites.append(Path(m.name).stem)
            lon.append(float(df["longitude"].iloc[0]))
            lat.append(float(df["latitude"].iloc[0]))
    np.savez(out, site=np.concatenate(s_all), t=np.concatenate(t_all), pol=np.concatenate(p_all),
             site_lon=np.array(lon), site_lat=np.array(lat), site_name=np.array(sites),
             pollutants=np.array(BRASOV_POLLUTANTS))
    print(f"[brasov] {len(sites)} sites, {sum(len(x) for x in t_all):,} readings -> {out}")
    return out


def _delhi_file_list(months):
    with urllib.request.urlopen(HF_TREE_URL, timeout=120) as r:
        tree = json.load(r)
    files = sorted(f["path"] for f in tree if f.get("type") == "file" and f["path"].startswith("1.Raw(PM)/"))
    return [p for p in files if p.split("/")[1] in months]


def prepare_delhi(data_dir=DATA_DIR, months=("Dec2020", "Jan2021")):
    """Build data/delhi.npz: one row per cleaned reading (device, time, lat, lon, PM2.5, PM10)."""
    data_dir = Path(data_dir)
    out = data_dir / "delhi.npz"
    if out.exists():
        print(f"[delhi] cache exists: {out}")
        return out
    paths = _delhi_file_list(set(months))
    if not paths:
        raise RuntimeError(f"no AirDelhi files found for {months}")
    raw_dir = data_dir / "raw" / "airdelhi"

    def fetch(p):
        return _download(HF_FILE_URL + urllib.parse.quote(p), raw_dir / p.split("/")[-1])

    with ThreadPoolExecutor(8) as ex:
        local = list(ex.map(fetch, paths))
    t0 = pd.Timestamp("2020-12-01 00:00:00", tz="Asia/Kolkata")
    frames = []
    for f in local:
        df = pd.read_csv(f, usecols=["dateTime", "deviceId", "lat", "long", "pm2_5", "pm10"])
        df["dateTime"] = pd.to_datetime(df["dateTime"], utc=True).dt.tz_convert("Asia/Kolkata")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    n_raw = len(df)
    ok = (df["lat"].between(*DELHI_BOX["lat"]) & df["long"].between(*DELHI_BOX["lon"])
          & (df["pm2_5"] > 0) & (df["pm2_5"] <= 1000) & (df["pm10"] > 0) & (df["pm10"] <= 2000))
    df = df[ok].drop_duplicates(["deviceId", "dateTime"]).sort_values("dateTime")
    devices = {d: i for i, d in enumerate(sorted(df["deviceId"].unique()))}
    t = (df["dateTime"] - t0).dt.total_seconds().to_numpy(np.int32)
    np.savez(out, device=df["deviceId"].map(devices).to_numpy(np.int16), t=t,
             lat=df["lat"].to_numpy(np.float32), lon=df["long"].to_numpy(np.float32),
             pol=df[DELHI_POLLUTANTS].to_numpy(np.float32), device_name=np.array(list(devices)),
             pollutants=np.array(DELHI_POLLUTANTS), t0=np.array(str(t0)))
    print(f"[delhi] {len(local)} daily files, {n_raw:,} raw -> {len(df):,} clean readings, "
          f"{len(devices)} devices -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["brasov", "delhi", "all"], default="all")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    args = ap.parse_args()
    cfg = load_config()
    if args.dataset in ("brasov", "all"):
        prepare_brasov(args.data_dir)
    if args.dataset in ("delhi", "all"):
        months = tuple(cfg["workload"]["delhi_train_months"] + cfg["workload"]["delhi_test_months"])
        prepare_delhi(args.data_dir, months)


if __name__ == "__main__":
    main()
