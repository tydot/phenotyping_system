import pandas as pd
from backend.config import BOUNDARY_THRESHOLD

def build_consensus(seed_df: pd.DataFrame) -> pd.DataFrame:
    if seed_df.empty:
        return pd.DataFrame(columns=["patient_id", "cluster", "confidence", "is_boundary"])

    vote = (
        seed_df.groupby(["patient_id", "cluster"])
        .size()
        .reset_index(name="count")
    )

    idx = vote.groupby("patient_id")["count"].idxmax()
    best = vote.loc[idx].copy()

    total = seed_df.groupby("patient_id").size().rename("total").reset_index()
    best = best.merge(total, on="patient_id", how="left")
    best["confidence"] = best["count"] / best["total"]
    best["is_boundary"] = best["confidence"] < BOUNDARY_THRESHOLD
    best = best.rename(columns={"cluster": "consensus_cluster"})
    return best[["patient_id", "consensus_cluster", "confidence", "is_boundary"]]

def get_seed_assignments(seed_df: pd.DataFrame, patient_id: str) -> dict:
    sub = seed_df[seed_df["patient_id"] == patient_id].sort_values("seed")
    return {int(r.seed): int(r.cluster) for r in sub.itertuples()}