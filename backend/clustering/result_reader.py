import json
import pandas as pd
from pathlib import Path
from backend.config import BASELINE_DIR, ATTN_DIR, DEFAULT_TOPK, DEFAULT_SEEDS

def load_baseline_clusters() -> pd.DataFrame:
    return pd.read_csv(BASELINE_DIR / "clusters.csv")

def load_baseline_metrics() -> dict:
    with open(BASELINE_DIR / "cluster_metrics.json", "r", encoding="utf-8") as f:
        return json.load(f)

def load_seed_clusters(topk: int = DEFAULT_TOPK, seeds=DEFAULT_SEEDS) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        fp = ATTN_DIR / f"topk{topk}" / f"seed{seed}" / "clusters.csv"
        if fp.exists():
            df = pd.read_csv(fp)
            df["seed"] = seed
            rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["patient_id", "cluster", "seed"])
    return pd.concat(rows, ignore_index=True)

def load_topk_details(topk: int = DEFAULT_TOPK) -> pd.DataFrame:
    fp = ATTN_DIR / f"topk{topk}" / "seed0" / "attn_topk_details.csv"
    if fp.exists():
        return pd.read_csv(fp)
    return pd.DataFrame()