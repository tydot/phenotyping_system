from typing import Optional, Dict, List
import hashlib
import hmac
import secrets

from backend.db.database import get_conn, init_db


def init_user_db():
    """
    初始化用户权限相关表
    """
    init_db()


def _hash_password(password: str, salt: str) -> str:
    pwd_bytes = password.encode("utf-8")
    salt_bytes = salt.encode("utf-8")
    hashed = hashlib.pbkdf2_hmac("sha256", pwd_bytes, salt_bytes, 100000)
    return hashed.hex()


def _verify_password(password: str, salt: str, password_hash: str) -> bool:
    input_hash = _hash_password(password, salt)
    return hmac.compare_digest(input_hash, password_hash)


def register_user(
    username: str,
    password: str,
    role: str = "patient",
    patient_id: Optional[str] = None,
    full_name: Optional[str] = None,
) -> Dict:
    """
    注册用户
    公开注册页面只建议注册 patient。
    admin / doctor 应由管理员或脚本创建。
    """
    username = (username or "").strip()
    password = password or ""
    role = (role or "").strip().lower()
    patient_id = (str(patient_id).strip() if patient_id is not None else None)
    full_name = (full_name or "").strip() or None

    if not username:
        return {"ok": False, "message": "用户名不能为空。"}

    if len(username) < 3:
        return {"ok": False, "message": "用户名至少 3 位。"}

    if len(password) < 6:
        return {"ok": False, "message": "密码至少 6 位。"}

    if role not in {"admin", "doctor", "patient"}:
        return {"ok": False, "message": "角色必须是 admin、doctor 或 patient。"}

    # 关键约束：患者账号必须绑定 patient_id
    if role == "patient" and not patient_id:
        return {"ok": False, "message": "患者账号注册时必须填写 patient_id。"}

    init_user_db()

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT id FROM users WHERE username = ?", (username,))
    exists = cur.fetchone()
    if exists:
        conn.close()
        return {"ok": False, "message": "用户名已存在。"}

    # 患者 patient_id 也不能重复绑定多个患者账号（可选但建议）
    if role == "patient":
        cur.execute("""
            SELECT id FROM users
            WHERE role = 'patient' AND patient_id = ?
            LIMIT 1
        """, (patient_id,))
        pid_exists = cur.fetchone()
        if pid_exists:
            conn.close()
            return {"ok": False, "message": "该 patient_id 已绑定其他患者账号。"}

    salt = secrets.token_hex(16)
    password_hash = _hash_password(password, salt)

    cur.execute("""
        INSERT INTO users (username, password_hash, salt, role, patient_id, full_name, is_active)
        VALUES (?, ?, ?, ?, ?, ?, 1)
    """, (username, password_hash, salt, role, patient_id, full_name))

    conn.commit()
    conn.close()

    return {"ok": True, "message": "注册成功，请登录。"}


def authenticate_user(username: str, password: str) -> Optional[Dict]:
    username = (username or "").strip()
    password = password or ""

    init_user_db()

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, username, password_hash, salt, role, patient_id, full_name, is_active, created_at
        FROM users
        WHERE username = ?
        LIMIT 1
    """, (username,))
    row = cur.fetchone()
    conn.close()

    if row is None:
        return None

    if not bool(row["is_active"]):
        return None

    if not _verify_password(password, row["salt"], row["password_hash"]):
        return None

    return {
        "id": row["id"],
        "username": row["username"],
        "role": row["role"],
        "patient_id": row["patient_id"],
        "full_name": row["full_name"],
        "is_active": row["is_active"],
        "created_at": row["created_at"],
    }


def get_user_by_username(username: str) -> Optional[Dict]:
    username = (username or "").strip()

    init_user_db()

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, username, role, patient_id, full_name, is_active, created_at
        FROM users
        WHERE username = ?
        LIMIT 1
    """, (username,))
    row = cur.fetchone()
    conn.close()

    if row is None:
        return None

    return {
        "id": row["id"],
        "username": row["username"],
        "role": row["role"],
        "patient_id": row["patient_id"],
        "full_name": row["full_name"],
        "is_active": row["is_active"],
        "created_at": row["created_at"],
    }


def get_user_by_id(user_id: int) -> Optional[Dict]:
    init_user_db()

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, username, role, patient_id, full_name, is_active, created_at
        FROM users
        WHERE id = ?
        LIMIT 1
    """, (user_id,))
    row = cur.fetchone()
    conn.close()

    if row is None:
        return None

    return {
        "id": row["id"],
        "username": row["username"],
        "role": row["role"],
        "patient_id": row["patient_id"],
        "full_name": row["full_name"],
        "is_active": row["is_active"],
        "created_at": row["created_at"],
    }


def assign_patient_to_doctor(doctor_user_id: int, patient_id: str) -> Dict:
    init_user_db()

    patient_id = (str(patient_id or "")).strip()
    if not patient_id:
        return {"ok": False, "message": "patient_id 不能为空。"}

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, role, is_active
        FROM users
        WHERE id = ?
        LIMIT 1
    """, (doctor_user_id,))
    doctor = cur.fetchone()

    if doctor is None:
        conn.close()
        return {"ok": False, "message": "医生用户不存在。"}

    if doctor["role"] != "doctor":
        conn.close()
        return {"ok": False, "message": "该用户不是医生角色。"}

    if not bool(doctor["is_active"]):
        conn.close()
        return {"ok": False, "message": "该医生账号已停用。"}

    cur.execute("""
        INSERT OR IGNORE INTO doctor_patient_map (doctor_user_id, patient_id)
        VALUES (?, ?)
    """, (doctor_user_id, patient_id))

    conn.commit()
    conn.close()
    return {"ok": True, "message": "绑定成功。"}


def remove_patient_from_doctor(doctor_user_id: int, patient_id: str) -> Dict:
    init_user_db()

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        DELETE FROM doctor_patient_map
        WHERE doctor_user_id = ? AND patient_id = ?
    """, (doctor_user_id, str(patient_id).strip()))
    conn.commit()
    conn.close()
    return {"ok": True, "message": "解绑成功。"}


def get_doctor_patient_ids(doctor_user_id: int) -> List[str]:
    init_user_db()

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT patient_id
        FROM doctor_patient_map
        WHERE doctor_user_id = ?
        ORDER BY patient_id
    """, (doctor_user_id,))
    rows = cur.fetchall()
    conn.close()

    return [str(r["patient_id"]) for r in rows]


def ensure_auth_state():
    import streamlit as st

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if "current_user" not in st.session_state:
        st.session_state.current_user = None


def login_user_session(user: Dict):
    import streamlit as st

    st.session_state.logged_in = True
    st.session_state.current_user = user


def logout_user_session():
    import streamlit as st

    st.session_state.logged_in = False
    st.session_state.current_user = None


def get_current_user() -> Optional[Dict]:
    import streamlit as st

    ensure_auth_state()
    return st.session_state.current_user


def require_login() -> Dict:
    import streamlit as st

    ensure_auth_state()
    if not st.session_state.logged_in or not st.session_state.current_user:
        st.warning("请先登录后再访问该页面。")
        st.page_link("pages/0_登录注册.py", label="前往登录页", icon="🔐")
        st.stop()

    return st.session_state.current_user


def require_role(*roles: str) -> Dict:
    import streamlit as st

    user = require_login()
    if user.get("role") not in roles:
        st.error("您无权限访问该页面。")
        st.stop()
    return user


def can_view_patient(user: Dict, patient_id: str) -> bool:
    """
    对象级权限控制：
    - admin: 可看全部
    - doctor: 只能看 doctor_patient_map 里绑定的患者
    - patient: 只能看自己的 patient_id
    """
    if not user or not patient_id:
        return False

    target_pid = str(patient_id).strip()
    role = user.get("role")

    if role == "admin":
        return True

    if role == "doctor":
        allowed = get_doctor_patient_ids(int(user["id"]))
        return target_pid in allowed

    if role == "patient":
        own_pid = str(user.get("patient_id") or "").strip()
        return own_pid == target_pid

    return False