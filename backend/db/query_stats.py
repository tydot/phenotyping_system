from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATS_DIR = PROJECT_ROOT / "outputs" / "statistics"

KRUSKAL_FILES = {
    "stable": STATS_DIR / "attn_pooling-8_kruskal_summary_stable.csv",
    "all": STATS_DIR / "attn_pooling-8_kruskal_summary_all.csv",
}

DUNN_FILES = {
    "stable": STATS_DIR / "attn_pooling-8_dunn_posthoc_stable.csv",
    "all": STATS_DIR / "attn_pooling-8_dunn_posthoc_all.csv",
}


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"未找到统计文件: {path}")
    return pd.read_csv(path)


def get_kruskal_summary(population: str = "stable") -> pd.DataFrame:
    if population not in KRUSKAL_FILES:
        raise ValueError(f"不支持的 population: {population}")
    return _load_csv(KRUSKAL_FILES[population])


def get_dunn_posthoc(population: str = "stable") -> pd.DataFrame:
    if population not in DUNN_FILES:
        raise ValueError(f"不支持的 population: {population}")
    return _load_csv(DUNN_FILES[population])


def get_available_metrics(population: str = "stable") -> list[str]:
    df = get_kruskal_summary(population)
    if "指标" not in df.columns:
        return []
    return df["指标"].dropna().astype(str).unique().tolist()


def get_metric_kruskal(metric: str, population: str = "stable") -> pd.DataFrame:
    df = get_kruskal_summary(population)
    if "指标" not in df.columns:
        return pd.DataFrame()
    return df[df["指标"].astype(str) == str(metric)].copy()


def get_metric_dunn(metric: str, population: str = "stable") -> pd.DataFrame:
    df = get_dunn_posthoc(population)
    if "指标" not in df.columns:
        return pd.DataFrame()
    return df[df["指标"].astype(str) == str(metric)].copy()