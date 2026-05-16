from backend.db.database import get_conn


def ensure_inference_table():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS inference_task (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            sex TEXT,
            age REAL,
            main_symptom TEXT,

            resting_pressure REAL,
            msp REAL,
            squeeze_duration REAL,
            defecatory_rectal_pressure REAL,
            first_sensation REAL,
            desire_to_defecate REAL,
            urgency_threshold REAL,
            max_tolerable_volume REAL,
            rair_min_volume REAL,
            anal_length REAL,

            arm_image_names TEXT,
            rair_image_names TEXT,

            predicted_cluster INTEGER,
            confidence REAL,
            is_boundary INTEGER,
            similar_cases TEXT,
            summary TEXT,
            model_version TEXT,
            inference_time TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_inference_result(payload: dict, result: dict, arm_image_names: str = "", rair_image_names: str = ""):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO inference_task (
            patient_id, sex, age, main_symptom,
            resting_pressure, msp, squeeze_duration, defecatory_rectal_pressure,
            first_sensation, desire_to_defecate, urgency_threshold, max_tolerable_volume,
            rair_min_volume, anal_length,
            arm_image_names, rair_image_names,
            predicted_cluster, confidence, is_boundary, similar_cases,
            summary, model_version, inference_time
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        payload.get("patient_id"),
        payload.get("sex"),
        payload.get("age"),
        payload.get("main_symptom"),

        payload.get("resting_pressure"),
        payload.get("msp"),
        payload.get("squeeze_duration"),
        payload.get("defecatory_rectal_pressure"),
        payload.get("first_sensation"),
        payload.get("desire_to_defecate"),
        payload.get("urgency_threshold"),
        payload.get("max_tolerable_volume"),
        payload.get("rair_min_volume"),
        payload.get("anal_length"),

        arm_image_names,
        rair_image_names,

        result.get("predicted_cluster"),
        result.get("confidence"),
        1 if result.get("is_boundary") else 0,
        ",".join(result.get("similar_cases", [])),
        result.get("summary"),
        result.get("model_version"),
        result.get("inference_time"),
    ))

    conn.commit()
    conn.close()