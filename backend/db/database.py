import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "app_data.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    # 用户表：支持管理员 / 医生 / 患者
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin', 'doctor', 'patient')),
            patient_id TEXT,
            full_name TEXT,
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 医生-患者映射表：医生只能看自己绑定的患者
    cur.execute("""
        CREATE TABLE IF NOT EXISTS doctor_patient_map (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doctor_user_id INTEGER NOT NULL,
            patient_id TEXT NOT NULL,
            UNIQUE(doctor_user_id, patient_id),
            FOREIGN KEY (doctor_user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()