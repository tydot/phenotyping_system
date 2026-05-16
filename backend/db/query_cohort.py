from backend.db.database import get_conn


def get_cohort_overview():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) AS n FROM patient_clinical")
    n_patients = cur.fetchone()["n"]

    cur.execute("SELECT COUNT(*) AS n FROM patient_clinical WHERE confidence >= 0.8")
    n_stable = cur.fetchone()["n"]

    cur.execute("SELECT COUNT(*) AS n FROM patient_clinical WHERE confidence < 0.8")
    n_boundary = cur.fetchone()["n"]

    cur.execute("""
        SELECT consensus_cluster, COUNT(*) AS n
        FROM patient_clinical
        GROUP BY consensus_cluster
        ORDER BY consensus_cluster
    """)
    cluster_dist = [dict(r) for r in cur.fetchall()]

    conn.close()

    return {
        "n_patients": n_patients,
        "n_stable": n_stable,
        "n_boundary": n_boundary,
        "cluster_dist": cluster_dist,
    }


def get_clinical_field_coverage():
    conn = get_conn()
    cur = conn.cursor()

    fields = [
        ("sex", "性别"),
        ("age", "年龄"),
        ("main_symptom", "主要症状"),
        ("resting_pressure", "肛门括约肌静息压"),
        ("msp", "最大缩榨压"),
        ("squeeze_duration", "缩肛持续时间"),
        ("defecatory_rectal_pressure", "排便时直肠压力"),
        ("first_sensation", "初始感觉阈值"),
        ("desire_to_defecate", "初始便意阈值"),
        ("urgency_threshold", "排便窘迫感阈值"),
        ("max_tolerable_volume", "最大容量感觉阈值"),
        ("rair_min_volume", "RAIR诱发最小容积"),
        ("anal_length", "肛门括约肌长度"),
    ]

    cur.execute("SELECT COUNT(*) AS n FROM patient_clinical")
    total = cur.fetchone()["n"]

    results = []
    for col, label in fields:
        cur.execute(f"SELECT COUNT(*) AS n FROM patient_clinical WHERE {col} IS NOT NULL")
        n_valid = cur.fetchone()["n"]
        results.append({
            "field": col,
            "label": label,
            "n_valid": n_valid,
            "coverage_rate": n_valid / total if total else 0.0,
        })

    conn.close()
    return results


def get_key_clinical_summary():
    conn = get_conn()
    cur = conn.cursor()

    fields = [
        ("resting_pressure", "肛门括约肌静息压"),
        ("msp", "最大缩榨压"),
        ("squeeze_duration", "缩肛持续时间"),
        ("defecatory_rectal_pressure", "排便时直肠压力"),
        ("first_sensation", "初始感觉阈值"),
        ("desire_to_defecate", "初始便意阈值"),
        ("urgency_threshold", "排便窘迫感阈值"),
        ("max_tolerable_volume", "最大容量感觉阈值"),
        ("rair_min_volume", "RAIR诱发最小容积"),
        ("anal_length", "肛门括约肌长度"),
    ]

    results = []
    for col, label in fields:
        cur.execute(f"""
            SELECT
                COUNT({col}) AS n_valid,
                AVG({col}) AS mean_value,
                MIN({col}) AS min_value,
                MAX({col}) AS max_value
            FROM patient_clinical
        """)
        row = dict(cur.fetchone())
        results.append({
            "field": col,
            "label": label,
            "n_valid": row["n_valid"],
            "mean_value": row["mean_value"],
            "min_value": row["min_value"],
            "max_value": row["max_value"],
        })

    conn.close()
    return results