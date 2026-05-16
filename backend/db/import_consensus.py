import pandas as pd
from pathlib import Path
from backend.db.database import get_conn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
from pathlib import Path
CONSENSUS_CSV = Path(r"G:\windows\图像数据\dataProcess\outputs\final_attn_tau007_topk8_pca100\topk8\consensus_labels.csv")

def import_patient_consensus():
    if not CONSENSUS_CSV.exists():
        raise FileNotFoundError(f"未找到文件: {CONSENSUS_CSV}")

    df = pd.read_csv(CONSENSUS_CSV)

    required_cols = [
        "patient_id",
        "pid_key",
        "consensus_cluster",
        "confidence",
        "switch_rate",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"consensus_labels.csv 缺少字段: {missing}")

    df["is_boundary"] = (df["confidence"] < 0.8).astype(int)

    # 全部转成字符串，避免患者编号被科学计数法/浮点污染
    df["patient_id"] = df["patient_id"].astype(str)
    df["pid_key"] = df["pid_key"].astype(str)

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("DELETE FROM patient_consensus")

    for _, row in df.iterrows():
        cur.execute("""
            INSERT INTO patient_consensus (
                patient_id,
                pid_key,
                consensus_cluster,
                confidence,
                switch_rate,
                is_boundary
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            row["patient_id"],
            row["pid_key"],
            int(row["consensus_cluster"]),
            float(row["confidence"]),
            float(row["switch_rate"]),
            int(row["is_boundary"]),
        ))

    conn.commit()
    conn.close()
    print(f"patient_consensus 导入完成，共 {len(df)} 条。")

if __name__ == "__main__":
    import_patient_consensus()