from backend.db.database import get_conn

def get_all_patient_consensus():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT patient_id, consensus_cluster, confidence, switch_rate, is_boundary
        FROM patient_consensus
    """)
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_boundary_patients():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT patient_id, consensus_cluster, confidence, switch_rate
        FROM patient_consensus
        WHERE is_boundary = 1
        ORDER BY confidence ASC
    """)
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]