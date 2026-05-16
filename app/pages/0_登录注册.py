import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from backend.auth.auth_service import (
    init_user_db,
    register_user,
    authenticate_user,
    login_user_session,
    logout_user_session,
    ensure_auth_state,
    get_current_user,
)

st.set_page_config(page_title="登录注册 | ARM 功能表型系统", layout="centered")

init_user_db()
ensure_auth_state()

st.title("🔐 登录注册")
st.caption("ARM 功能表型系统 - 账号访问入口")

current_user = get_current_user()

if current_user:
    st.success(
        f"当前已登录：{current_user.get('username')} | "
        f"角色：{current_user.get('role')} | "
        f"patient_id：{current_user.get('patient_id') or '-'}"
    )

    if st.button("退出登录", use_container_width=True):
        logout_user_session()
        st.rerun()

    st.divider()
    st.info(
        "权限说明：\n\n"
        "- 患者账号：只能查看自己的报告与自己的患者页\n"
        "- 医生账号：可查看绑定患者，访问 cohort / stability / upload inference\n"
        "- 管理员账号：可访问全部页面"
    )
    st.stop()

tab1, tab2 = st.tabs(["登录", "患者注册"])

with tab1:
    st.subheader("用户登录")
    st.caption("管理员、医生、已存在患者账号，请直接登录，不要重复注册。")

    login_username = st.text_input("用户名", key="login_username")
    login_password = st.text_input("密码", type="password", key="login_password")

    if st.button("登录", use_container_width=True, key="login_btn"):
        if not login_username.strip() or not login_password:
            st.error("请输入用户名和密码。")
            st.stop()

        user = authenticate_user(login_username, login_password)
        if not user:
            st.error("用户名或密码错误。")
        else:
            login_user_session(user)
            st.success("登录成功。")
            st.rerun()

with tab2:
    st.subheader("新患者首次注册")
    st.caption(
        "这里只用于“系统中还没有账号的新患者”首次建号。\n\n"
        "如果你是医生、管理员，或该患者已经有账号，请直接去“登录”，不要重复注册。"
    )

    reg_full_name = st.text_input("姓名", key="reg_full_name")
    reg_username = st.text_input("用户名", key="reg_username")
    reg_patient_id = st.text_input(
        "绑定患者ID（patient_id）",
        key="reg_patient_id",
        placeholder="例如：210259070"
    )
    reg_password = st.text_input("密码", type="password", key="reg_password")
    reg_password2 = st.text_input("确认密码", type="password", key="reg_password2")

    if st.button("注册患者账号", use_container_width=True, key="register_btn"):
        if not reg_username.strip():
            st.error("用户名不能为空。")
            st.stop()

        if not reg_patient_id.strip():
            st.error("患者账号必须填写 patient_id。")
            st.stop()

        if reg_password != reg_password2:
            st.error("两次输入的密码不一致。")
            st.stop()

        result = register_user(
            username=reg_username.strip(),
            password=reg_password,
            role="patient",
            patient_id=reg_patient_id.strip(),
            full_name=reg_full_name.strip() or None,
        )

        if result.get("ok"):
            st.success(result.get("message", "注册成功，请登录。"))
        else:
            msg = result.get("message", "注册失败。")
            st.error(msg)

            if "已绑定其他患者账号" in msg:
                st.info(
                    "这个 patient_id 已经有账号了。\n\n"
                    "请直接使用原有账号登录；如果忘记密码，请联系管理员重置密码。"
                )

st.divider()
st.caption("⚠️ 患者注册时请确保 patient_id 与系统内患者编号一致，否则将无法查看对应报告。")