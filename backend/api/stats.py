
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.auth.auth_service import require_login

require_login()
from backend.db.query_stats import (
    get_kruskal_summary,
    get_dunn_posthoc,
    get_available_metrics,
)


def _ensure_df(x):
    if isinstance(x, pd.DataFrame):
        return x.copy()
    try:
        return pd.DataFrame(x or [])
    except Exception:
        return pd.DataFrame()


def get_stats_view(population: str = "stable", metric: str | None = None):
    population = population if population in {"stable", "all"} else "stable"

    kruskal_df = _ensure_df(get_kruskal_summary(population))
    dunn_df = _ensure_df(get_dunn_posthoc(population))
    metrics = list(get_available_metrics(population) or [])

    if metric and metric != "全部指标":
        if "指标" in kruskal_df.columns:
            kruskal_df = kruskal_df[kruskal_df["指标"].astype(str) == str(metric)].copy()
        if "指标" in dunn_df.columns:
            dunn_df = dunn_df[dunn_df["指标"].astype(str) == str(metric)].copy()

    sig_count = 0
    if "是否显著_adj" in kruskal_df.columns:
        sig_count = int((kruskal_df["是否显著_adj"] == True).sum())
    elif "显著性_adj" in kruskal_df.columns:
        sig_count = int((kruskal_df["显著性_adj"].astype(str) != "ns").sum())

    summary_text = (
        f"当前展示的是 {population} 人群的 Kruskal–Wallis 与 Dunn 事后检验结果。"
        f"共纳入 {len(kruskal_df)} 个指标，其中显著指标 {sig_count} 个。"
    )

    return {
        "population": population,
        "available_metrics": ["全部指标"] + metrics,
        "kruskal_summary": kruskal_df.to_dict(orient="records"),
        "dunn_posthoc": dunn_df.to_dict(orient="records"),
        "summary_text": summary_text,
    }
