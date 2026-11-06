"""Configuration loading and the named rule variants used in the experiments."""
import copy
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "default.toml"
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"

# Server-side rule sets. "baseline" is EdgePlus as published (first come first served, full speed).
VARIANTS = {
    "baseline":       {"queue": "fcfs", "late": "keep", "dvfs": "off"},
    "fcfs_drop":      {"queue": "fcfs", "late": "drop", "dvfs": "off"},
    "edf_keep":       {"queue": "edf",  "late": "keep", "dvfs": "off"},
    "class_priority": {"queue": "prio", "late": "drop", "dvfs": "off"},
    "edf_drop":       {"queue": "edf",  "late": "drop", "dvfs": "off"},
    "naive_dvfs":     {"queue": "fcfs", "late": "keep", "dvfs": "naive"},
    "guarded_dvfs":   {"queue": "fcfs", "late": "keep", "dvfs": "guarded"},
    "full":           {"queue": "edf",  "late": "drop", "dvfs": "guarded"},
}


def load_config(path=None, overrides=None):
    """Read the TOML config. `overrides` maps "section.key" to a value."""
    with open(path or DEFAULT_CONFIG, "rb") as f:
        cfg = tomllib.load(f)
    for key, value in (overrides or {}).items():
        section, name = key.split(".", 1)
        cfg[section][name] = value
    return cfg


def with_rules(cfg, variant):
    """Return a copy of cfg whose [rules] section is set to the named variant."""
    out = copy.deepcopy(cfg)
    out["rules"].update(VARIANTS[variant])
    return out
