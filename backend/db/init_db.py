from backend.db.database import get_conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS patient_consensus (
        patient_id TEXT PRIMARY KEY,
        pid_key TEXT,
        consensus_cluster INTEGER NOT NULL,
        confidence REAL NOT NULL,
        switch_rate REAL NOT NULL,
        is_boundary INTEGER NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS patient_clinical (
        patient_id TEXT PRIMARY KEY,
        pid_key TEXT,
        consensus_cluster INTEGER,
        confidence REAL,
        switch_rate REAL,

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
        anal_length REAL
    )
    """)

    conn.commit()
    conn.close()
    print("数据库初始化完成。")

if __name__ == "__main__":
    init_db()