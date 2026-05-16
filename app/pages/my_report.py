import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np

from backend.auth.auth_service import require_role
from backend.api.patient import get_patient_view
from backend.version_manager import (
    load_versions_config,
    get_default_version_key,
    get_version_config,
    get_version_files,
    resolve_path,
    safe_read_csv,
)


st.set_page_config(page_title="我的报告 | ARM 功能表型系统", layout="wide")


# ============================================================
# 工具函数
# ============================================================

def fmt_number(x, digits=2):
    if x is None or x == "":
        return "-"
    if isinstance(x, bool):
        return "是" if x else "否"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def find_first_existing_col(df: pd.DataFrame, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalize_bool_value(x):
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in ["true", "1", "yes", "y", "是", "边界", "boundary"]


@st.cache_data(show_spinner=False)
def load_version_csv(file_path_str: str):
    path = resolve_path(file_path_str)
    return safe_read_csv(path)


def get_patient_version_row(patient_id: str, current_files: dict, boundary_threshold: float):
    """
    患者端默认从当前 default_version 的结果文件中查找患者分型结果。
    优先读取 clinical_with_clusters，其次读取 consensus_labels。
    """
    candidate_files = [
        current_files.get("clinical_with_clusters"),
        current_files.get("consensus_labels"),
    ]

    for file_path in candidate_files:
        if not file_path:
            continue

        df = load_version_csv(file_path)
        if df is None or df.empty:
            continue

        patient_col = find_first_existing_col(
            df,
            [
                "patient_id",
                "patient",
                "PatientID",
                "Patient_ID",
                "id",
                "病例号",
                "患者编号",
            ],
        )

        if not patient_col:
            continue

        df = df.copy()
        df[patient_col] = df[patient_col].astype(str).str.strip()
        row_df = df[df[patient_col] == str(patient_id).strip()]

        if row_df.empty:
            continue

        row = row_df.iloc[0].to_dict()

        cluster_col = find_first_existing_col(
            df,
            [
                "consensus_cluster",
                "cluster",
                "Cluster",
                "final_cluster",
                "label",
                "consensus_label",
                "共识簇",
                "AI分型",
            ],
        )

        confidence_col = find_first_existing_col(
            df,
            [
                "confidence",
                "Confidence",
                "consensus_confidence",
                "vote_confidence",
                "stability_confidence",
                "置信度",
                "稳定性置信度",
            ],
        )

        boundary_col = find_first_existing_col(
            df,
            [
                "is_boundary",
                "boundary",
                "Is_Boundary",
                "边界患者",
                "是否边界",
            ],
        )

        cluster_val = row.get(cluster_col) if cluster_col else None

        confidence_val = None
        if confidence_col:
            try:
                confidence_val = float(row.get(confidence_col))
            except Exception:
                confidence_val = None

        if boundary_col:
            is_boundary_val = normalize_bool_value(row.get(boundary_col))
        else:
            is_boundary_val = (
                confidence_val is not None and confidence_val < boundary_threshold
            )

        return {
            "raw_row": row,
            "source_file": file_path,
            "cluster": cluster_val,
            "confidence": confidence_val,
            "is_boundary": is_boundary_val,
        }

    return None


def safe_dict(value):
    return value if isinstance(value, dict) else {}


# ============================================================
# 权限控制
# ============================================================

user = require_role("patient")

patient_id = str(user.get("patient_id") or "").strip()
if not patient_id:
    st.error("当前患者账号未绑定 patient_id。")
    st.stop()


# ============================================================
# 默认版本配置：患者端只展示 default_version
# ============================================================

try:
    version_config = load_versions_config()
    default_version_key = get_default_version_key(version_config)
    current_version = get_version_config(default_version_key, version_config)
    current_files = get_version_files(default_version_key, version_config)
except Exception as e:
    st.error(f"版本配置读取失败：{e}")
    st.stop()

boundary_threshold = current_version.get("boundary_threshold", 0.8)
try:
    boundary_threshold = float(boundary_threshold)
except Exception:
    boundary_threshold = 0.8


# ============================================================
# 读取患者后端数据
# ============================================================

patient = get_patient_view(patient_id)
if not patient:
    st.error("未找到您的报告。")
    st.stop()

ai = safe_dict(patient.get("ai_result"))
phys = safe_dict(patient.get("physiology"))
rair = safe_dict(patient.get("rair"))


# ============================================================
# 用 default_version 结果覆盖 AI 分型结果
# ============================================================

version_patient_result = get_patient_version_row(
    patient_id=patient_id,
    current_files=current_files,
    boundary_threshold=boundary_threshold,
)

if version_patient_result:
    ai["cluster"] = version_patient_result.get("cluster")
    ai["confidence"] = version_patient_result.get("confidence")
    ai["is_boundary"] = version_patient_result.get("is_boundary")


# ============================================================
# 页面标题
# ============================================================

st.title("🧾 我的检查报告")
st.caption(f"患者ID：{patient_id}")

st.info(
    f"当前报告采用系统默认分型版本：**{current_version.get('display_name', default_version_key)}**  \n"
    f"方法配置：**{current_version.get('method', '-')}**  \n\n"
    "本报告仅用于科研与功能表型分析，不作为临床诊断或治疗决策依据。"
)

with st.expander("查看报告版本信息"):
    st.write("默认版本 key：", default_version_key)
    st.write("版本名称：", current_version.get("display_name", "-"))
    st.write("方法配置：", current_version.get("method", "-"))
    if version_patient_result:
        st.write("患者分型来源文件：", version_patient_result.get("source_file"))
    else:
        st.write("患者分型来源文件：未在当前版本结果文件中找到，使用后端默认结果。")


# ============================================================
# AI 分型结果
# ============================================================

st.subheader("AI 分型结果")

c1, c2, c3 = st.columns(3)

with c1:
    cluster_val = ai.get("cluster")
    st.metric("AI 分型", f"Cluster {cluster_val}" if cluster_val is not None else "未知")

with c2:
    confidence_val = ai.get("confidence")
    st.metric("稳定性置信度", fmt_number(confidence_val, 3))

with c3:
    st.metric("边界患者", "是" if ai.get("is_boundary", False) else "否")

if ai.get("is_boundary", False):
    st.warning(
        "该患者处于分型边界区域，说明其功能表型在不同聚类初始化下可能存在一定不稳定性。"
    )
else:
    st.success("该患者在当前默认版本下属于稳定分型患者。")

st.caption(
    f"当前系统采用 confidence ≥ {boundary_threshold} 视为稳定患者，"
    f"confidence < {boundary_threshold} 视为边界患者。"
)

st.divider()


# ============================================================
# 核心功能指标
# ============================================================

st.subheader("核心功能指标")

core_metrics = phys.get("core_metrics", {})

if core_metrics:
    df_core = pd.DataFrame(list(core_metrics.items()), columns=["指标", "数值"])
    df_core["数值"] = df_core["数值"].apply(lambda x: fmt_number(x, 3))
    st.dataframe(df_core, use_container_width=True, hide_index=True)
else:
    st.caption("暂无核心功能指标。")

st.divider()


# ============================================================
# RAIR 生理反射证据
# ============================================================

st.subheader("RAIR 生理反射证据")

time_series = rair.get("time_series")

if time_series is None:
    st.caption("暂无 RAIR 时间序列数据。")
else:
    try:
        ts = np.array(time_series)
        if ts.size == 0:
            st.caption("RAIR 时间序列为空。")
        else:
            st.line_chart(ts, height=220)
    except Exception:
        st.caption("RAIR 时间序列暂时无法可视化。")

features = rair.get("features", {})

if isinstance(features, dict) and features and features.get("available", True):
    df_rair = pd.DataFrame(list(features.items()), columns=["特征", "数值"])
    df_rair["数值"] = df_rair["数值"].apply(lambda x: fmt_number(x, 3))
    st.dataframe(df_rair, use_container_width=True, hide_index=True)
else:
    st.caption("暂无 RAIR 特征。")

st.divider()


# ============================================================
# 科研声明
# ============================================================

st.caption(
    "⚠️ 本页面仅展示患者本人在系统默认版本下的功能表型分析结果，"
    "不用于临床诊断、治疗建议或医疗决策。"
)