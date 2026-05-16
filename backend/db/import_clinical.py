import pandas as pd
from pathlib import Path
from backend.db.database import get_conn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
from pathlib import Path
CLINICAL_CSV = Path(r"G:\windows\图像数据\dataProcess\outputs\clinical_consensus_analysis\clinical_with_consensus_attn_topk8.csv")

def norm_pid(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s

def val(row, col):
    if col not in row or pd.isna(row[col]):
        return None
    return row[col]

def import_patient_clinical():
    if not CLINICAL_CSV.exists():
        raise FileNotFoundError(f"未找到文件: {CLINICAL_CSV}")

    df = pd.read_csv(CLINICAL_CSV)

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("DELETE FROM patient_clinical")

    for _, row in df.iterrows():
        patient_id = norm_pid(val(row, "病人编号"))
        pid_key = norm_pid(val(row, "pid_key"))

        cur.execute("""
            INSERT OR REPLACE INTO patient_clinical (
                patient_id, pid_key, consensus_cluster, confidence, switch_rate,
                sex, age, main_symptom,
                resting_pressure, msp, squeeze_duration, defecatory_rectal_pressure,
                first_sensation, desire_to_defecate, urgency_threshold,
                max_tolerable_volume, rair_min_volume, anal_length
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            patient_id,
            pid_key,
            val(row, "consensus_cluster"),
            val(row, "confidence"),
            val(row, "switch_rate"),

            val(row, "性别"),
            val(row, "年龄"),
            val(row, "主要症状"),

            val(row, "肛门括约肌静息压(mmHg)"),
            val(row, "最大缩榨压MSP（mmHg）"),
            val(row, "缩肛持续时间(s)"),
            val(row, "排便时直肠压力(mmHg)"),

            val(row, "初始感觉阈值(ml)"),
            val(row, "初始便意阈值(ml)"),
            val(row, "排便窘迫感阈值(ml)"),
            val(row, "最大容量感觉阈值(ml)"),
            val(row, "RAIR诱发最小容积(ml)"),
            val(row, "肛门括约肌长度(cm)")
        ))

    conn.commit()
    conn.close()
    print(f"patient_clinical 导入完成，共 {len(df)} 条。")

if __name__ == "__main__":
    import_patient_clinical()