from backend.db.database import get_conn


def normalize_pid(x):
    if x is None:
        return None
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def normalize_gender(value):
    if value is None:
        return None

    s = str(value).strip()
    if s == "":
        return None

    if s in ["男", "male", "Male", "M", "m", "1"]:
        return "male"
    if s in ["女", "female", "Female", "F", "f", "0"]:
        return "female"

    return s


def enrich_gender_fields(data: dict):
    """
    兼容 sex / gender / 性别 三种字段来源，
    并统一补充 gender 字段，保留原字段不删除。
    """
    if not data:
        return data

    raw_gender = (
        data.get("gender")
        or data.get("sex")
        or data.get("性别")
    )

    normalized = normalize_gender(raw_gender)
    data["gender"] = normalized

    # 可选：如果你也想让 sex 始终可用，就一并补齐
    if "sex" not in data or data.get("sex") in (None, ""):
        data["sex"] = raw_gender

    return data


def get_patient_consensus(patient_id: str):
    patient_id = normalize_pid(patient_id)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT patient_id, pid_key, consensus_cluster, confidence, switch_rate, is_boundary
        FROM patient_consensus
        WHERE patient_id = ?
    """, (patient_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    data = dict(row)

    # consensus 表一般没有 gender，但这里保留兼容逻辑，防止后续扩表
    data = enrich_gender_fields(data)
    return data


def get_patient_clinical(patient_id: str):
    patient_id = normalize_pid(patient_id)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT *
        FROM patient_clinical
        WHERE patient_id = ?
    """, (patient_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    data = dict(row)
    data = enrich_gender_fields(data)
    return data