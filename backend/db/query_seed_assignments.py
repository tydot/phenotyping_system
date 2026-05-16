
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

SEED_ASSIGNMENTS_CSV = DATA_DIR / "stability" / "seed_assignments_long.csv"

def normalize_pid(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s

def get_patient_seed_assignments(patient_id: str):
    if not SEED_ASSIGNMENTS_CSV.exists():
        return {}

    df = pd.read_csv(SEED_ASSIGNMENTS_CSV)

    required_cols = ["patient_id", "seed", "aligned_cluster"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"seed_assignments_long.csv 缺少字段: {missing}")

    df["patient_id"] = df["patient_id"].apply(normalize_pid)
    target_pid = normalize_pid(patient_id)

    sub = df[df["patient_id"] == target_pid].copy()
    if sub.empty:
        return {}

    sub = sub.sort_values("seed")
    return {int(r["seed"]): int(r["aligned_cluster"]) for _, r in sub.iterrows()}