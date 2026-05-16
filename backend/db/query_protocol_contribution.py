from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

PROTOCOL_CONTRIB_CSV = DATA_DIR / "protocol" / "protocol_contribution_norm.csv"
ATTN_TOPK_DETAILS_CSV = DATA_DIR / "protocol" / "attn_topk_details.csv"

PROTOCOL_ORDER = [
    "Contraction",
    "Cough",
    "Defecation",
    "RestPressure",
    "rair",
]


def normalize_pid(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s or None


def get_patient_protocol_contribution(patient_id: str):
    if not PROTOCOL_CONTRIB_CSV.exists():
        return {
            "available": False,
            "message": f"未找到协议贡献文件：{PROTOCOL_CONTRIB_CSV.name}",
        }

    df = pd.read_csv(PROTOCOL_CONTRIB_CSV)

    required_cols = ["patient_id", "protocol", "contribution"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"protocol_contribution_norm.csv 缺少字段: {missing}")

    df["patient_id"] = df["patient_id"].apply(normalize_pid)
    df["protocol"] = df["protocol"].astype(str).str.strip()
    df["contribution"] = pd.to_numeric(df["contribution"], errors="coerce")

    target_pid = normalize_pid(patient_id)
    sub = df[df["patient_id"] == target_pid].copy()

    if sub.empty:
        return {
            "available": False,
            "message": "当前患者暂无协议贡献结果。",
        }

    contrib_map = {
        row["protocol"]: float(row["contribution"])
        for _, row in sub.iterrows()
        if pd.notna(row["contribution"])
    }

    ordered = {}
    for p in PROTOCOL_ORDER:
        if p in contrib_map:
            ordered[p] = contrib_map[p]

    for p, v in contrib_map.items():
        if p not in ordered:
            ordered[p] = v

    return ordered


def get_patient_protocol_topk_details(patient_id: str, per_protocol_topn: int = 3):
    if not ATTN_TOPK_DETAILS_CSV.exists():
        return []

    df = pd.read_csv(ATTN_TOPK_DETAILS_CSV)

    required_cols = ["patient_id", "protocol", "rank", "filepath", "score", "weight"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"attn_topk_details.csv 缺少字段: {missing}")

    df["patient_id"] = df["patient_id"].apply(normalize_pid)
    df["protocol"] = df["protocol"].astype(str).str.strip()
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")

    target_pid = normalize_pid(patient_id)
    sub = df[df["patient_id"] == target_pid].copy()
    if sub.empty:
        return []

    sub["protocol_order"] = (
        sub["protocol"].map({p: i for i, p in enumerate(PROTOCOL_ORDER)}).fillna(999)
    )
    sub = sub.sort_values(
        ["protocol_order", "protocol", "rank"],
        ascending=[True, True, True],
    )

    rows = []
    for protocol, grp in sub.groupby("protocol", sort=False):
        grp = grp.sort_values(["rank", "weight"], ascending=[True, False]).head(per_protocol_topn)
        for _, row in grp.iterrows():
            filepath = str(row["filepath"]) if pd.notna(row["filepath"]) else ""
            rows.append(
                {
                    "protocol": protocol,
                    "rank": int(row["rank"]) if pd.notna(row["rank"]) else None,
                    "filepath": filepath,
                    "filename": Path(filepath).name if filepath else "",
                    "score": float(row["score"]) if pd.notna(row["score"]) else None,
                    "weight": float(row["weight"]) if pd.notna(row["weight"]) else None,
                }
            )

    return rows