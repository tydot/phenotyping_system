import sys
from pathlib import Path
import hashlib
import secrets

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.db.database import get_conn
from backend.auth.auth_service import init_user_db


def _hash_password(password: str, salt: str) -> str:
    pwd_bytes = password.encode("utf-8")
    salt_bytes = salt.encode("utf-8")
    hashed = hashlib.pbkdf2_hmac("sha256", pwd_bytes, salt_bytes, 100000)
    return hashed.hex()


init_user_db()

patient_id = input("请输入 patient_id: ").strip()
new_password = input("请输入新密码: ").strip()

conn = get_conn()
cur = conn.cursor()

cur.execute("""
    SELECT id, username, patient_id, role
    FROM users
    WHERE role = 'patient' AND patient_id = ?
    LIMIT 1
""", (patient_id,))
row = cur.fetchone()

if not row:
    print("未找到绑定该 patient_id 的患者账号。")
    conn.close()
    raise SystemExit

salt = secrets.token_hex(16)
password_hash = _hash_password(new_password, salt)

cur.execute("""
    UPDATE users
    SET password_hash = ?, salt = ?
    WHERE id = ?
""", (password_hash, salt, row["id"]))

conn.commit()
conn.close()

print(f"已重置账号密码：username={row['username']} | patient_id={row['patient_id']}")