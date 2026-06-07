import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st

from backend.api.inference import run_inference
from backend.db.query_inference import ensure_inference_table, save_inference_result
from backend.auth.auth_service import require_role


st.set_page_config(
    page_title="新病例上传推理 | ARM 功能表型系统",
    layout="wide"
)

# 只有管理员和医生可访问
user = require_role("admin", "doctor")

st.title("🧪 新病例上传与分型推理")
st.caption(
    f"基于 DINOv2 + 协议内 attention pooling + 现有 consensus cluster 原型映射的在线推理｜当前用户：{user.get('username', '-')}"
)

ensure_inference_table()

st.divider()

# ============================================================
# 基本信息
# ============================================================

st.subheader("1. 患者基本信息")

c1, c2, c3, c4 = st.columns(4)
with c1:
    patient_id = st.text_input("患者ID", value="")
with c2:
    sex = st.selectbox("性别", ["", "男", "女"])
with c3:
    age = st.number_input("年龄", min_value=0, max_value=120, value=0)
with c4:
    main_symptom = st.text_input("主要症状", value="")

st.divider()

# ============================================================
# 临床指标
# ============================================================

st.subheader("2. 临床指标录入")

c1, c2, c3 = st.columns(3)
with c1:
    resting_pressure = st.number_input("肛门括约肌静息压", value=0.0)
    msp = st.number_input("最大缩榨压（MSP）", value=0.0)
    squeeze_duration = st.number_input("缩肛持续时间", value=0.0)
    defecatory_rectal_pressure = st.number_input("排便时直肠压力", value=0.0)

with c2:
    first_sensation = st.number_input("初始感觉阈值", value=0.0)
    desire_to_defecate = st.number_input("初始便意阈值", value=0.0)
    urgency_threshold = st.number_input("排便窘迫感阈值", value=0.0)

with c3:
    max_tolerable_volume = st.number_input("最大容量感觉阈值", value=0.0)
    rair_min_volume = st.number_input("RAIR诱发最小容积", value=0.0)
    anal_length = st.number_input("肛门括约肌长度", value=0.0)

st.divider()

# ============================================================
# 5 协议上传
# ============================================================

st.subheader("3. 协议图像上传")
st.markdown("请按训练阶段一致的协议类型上传图像。")

u1, u2, u3, u4, u5 = st.columns(5)

with u1:
    contraction_files = st.file_uploader(
        "Contraction",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        key="contraction_files"
    )

with u2:
    cough_files = st.file_uploader(
        "Cough",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        key="cough_files"
    )

with u3:
    defecation_files = st.file_uploader(
        "Defecation",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        key="defecation_files"
    )

with u4:
    restpressure_files = st.file_uploader(
        "RestPressure",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        key="restpressure_files"
    )

with u5:
    rair_files = st.file_uploader(
        "RAIR",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        key="rair_files"
    )

protocol_files = {
    "Contraction": contraction_files or [],
    "Cough": cough_files or [],
    "Defecation": defecation_files or [],
    "RestPressure": restpressure_files or [],
    "rair": rair_files or [],
}

st.markdown("**当前上传概况**")
count_cols = st.columns(5)
for i, proto in enumerate(["Contraction", "Cough", "Defecation", "RestPressure", "rair"]):
    with count_cols[i]:
        st.metric(proto, len(protocol_files[proto]))

st.divider()

# ============================================================
# 推理按钮
# ============================================================

st.subheader("4. 开始分析")


def normalize_number(x):
    try:
        x = float(x)
        if x == 0:
            return None
        return x
    except Exception:
        return None


payload = {
    "patient_id": patient_id.strip(),
    "sex": sex.strip() if sex else "",
    "age": normalize_number(age),
    "main_symptom": main_symptom.strip(),
    "resting_pressure": normalize_number(resting_pressure),
    "msp": normalize_number(msp),
    "squeeze_duration": normalize_number(squeeze_duration),
    "defecatory_rectal_pressure": normalize_number(defecatory_rectal_pressure),
    "first_sensation": normalize_number(first_sensation),
    "desire_to_defecate": normalize_number(desire_to_defecate),
    "urgency_threshold": normalize_number(urgency_threshold),
    "max_tolerable_volume": normalize_number(max_tolerable_volume),
    "rair_min_volume": normalize_number(rair_min_volume),
    "anal_length": normalize_number(anal_length),
}

if st.button("开始分型分析", use_container_width=True):
    if not payload["patient_id"]:
        st.error("请先填写患者ID。")
        st.stop()

    total_uploaded = sum(len(v) for v in protocol_files.values())
    if total_uploaded == 0:
        st.error("请至少上传一个协议下的图像后再开始分析。")
        st.stop()

    with st.spinner("正在进行 DINOv2 特征提取与 patient-level 推理..."):
        response = run_inference(payload, protocol_files)

    if not response or not response.get("ok", False):
        st.error("当前输入不满足推理条件或推理失败。")
        for err in (response or {}).get("errors", ["未知错误"]):
            st.write(f"- {err}")
    else:
        result = response.get("result", {})

        all_image_names = []
        for _, files in protocol_files.items():
            all_image_names.extend([f.name for f in files])

        save_inference_result(
            payload=payload,
            result=result,
            arm_image_names=",".join(all_image_names),
            rair_image_names=",".join([f.name for f in protocol_files["rair"]]),
        )

        st.success("分析完成。")
        st.divider()

        st.subheader("5. 分型结果")

        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("预测 Cluster", result.get("predicted_cluster", "-"))
        with r2:
            confidence = result.get("confidence")
            if confidence is None:
                conf_display = "-"
            else:
                try:
                    conf_display = f"{float(confidence):.2%}"
                except Exception:
                    conf_display = str(confidence)
            st.metric("Confidence", conf_display)
        with r3:
            st.metric("边界病例", "是" if result.get("is_boundary", False) else "否")

        st.markdown("**结果摘要**")
        st.info(result.get("summary", "暂无摘要。"))

        st.markdown("**相似病例**")
        similar_cases = result.get("similar_cases", [])
        if similar_cases:
            for pid in similar_cases:
                st.write(f"- {pid}")
        else:
            st.write("暂无相似病例。")

        st.markdown("**参与推理的协议**")
        st.write(", ".join(result.get("protocols_used", [])) if result.get("protocols_used") else "无")

        st.markdown("**协议内注意力明细**")
        detail_rows = result.get("protocol_attention_details", [])
        if detail_rows:
            df_detail = pd.DataFrame(detail_rows)
            st.dataframe(df_detail, use_container_width=True, hide_index=True)
        else:
            st.write("暂无 attention 明细。")

        st.markdown("**推理元信息**")
        st.write(f"模型版本：{result.get('model_version', '-')}")
        st.write(f"分析时间：{result.get('inference_time', '-')}")

st.caption("⚠️ 在线阶段使用离线训练得到的 consensus cluster 原型进行映射；匈牙利对齐与投票属于离线建标签流程。")