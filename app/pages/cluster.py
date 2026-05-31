"""
集群视图（Cluster View）
ARM 功能表型系统
"""

import sys
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st

from backend.auth.auth_service import require_role
from backend.version_manager import (
    select_version_sidebar,
    safe_read_csv,
    resolve_path,
)


st.set_page_config(
    page_title="集群视图 | ARM 功能表型系统",
    layout="wide"
)


# ============================================================
# 权限控制
# ============================================================

user = require_role("admin", "doctor")


# ============================================================
# 版本选择
# ============================================================

selected_version, current_version, current_files = select_version_sidebar(
    key="cluster_selected_version"
)


# ============================================================
# 工具函数
# ============================================================

def fmt_ratio(x):
    try:
        return f"{float(x):.1%}"
    except Exception:
        return "-"


def fmt_score(x):
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "-"


def fmt_number(x, digits=2, suffix=""):
    try:
        return f"{float(x):.{digits}f}{suffix}"
    except Exception:
        return "-"


def find_first_existing_col(df: pd.DataFrame, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalize_cluster_df(df: pd.DataFrame, boundary_threshold: float) -> pd.DataFrame:
    """
    兼容不同 clinical_with_clusters.csv / merged_clinical.csv 的列名。
    统一生成：
    - patient_id
    - consensus_cluster
    - confidence
    - is_boundary
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    patient_col = find_first_existing_col(
    df,
    [
        "patient_id",
        "pid_key",
        "pid",
        "PID",
        "patient",
        "PatientID",
        "Patient_ID",
        "id",
        "病例号",
        "患者编号",
        "患者ID",
    ],
)

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
        ["is_boundary", "boundary", "Is_Boundary", "边界患者", "是否边界"],
    )

    rename_map = {}

    if patient_col:
        rename_map[patient_col] = "patient_id"
    if cluster_col:
        rename_map[cluster_col] = "consensus_cluster"
    if confidence_col:
        rename_map[confidence_col] = "confidence"
    if boundary_col:
        rename_map[boundary_col] = "is_boundary"

    df = df.rename(columns=rename_map)

    if "patient_id" not in df.columns:
        df["patient_id"] = df.index.astype(str)

    if "consensus_cluster" not in df.columns:
        df["consensus_cluster"] = None

    if "confidence" in df.columns:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    else:
        df["confidence"] = None

    if "is_boundary" not in df.columns:
        if "confidence" in df.columns and df["confidence"].notna().any():
            df["is_boundary"] = df["confidence"].apply(
                lambda x: bool(pd.notnull(x) and float(x) < boundary_threshold)
            )
        else:
            df["is_boundary"] = pd.NA
    else:
        def normalize_bool(x):
            if isinstance(x, bool):
                return x
            s = str(x).strip().lower()
            return s in ["true", "1", "yes", "y", "是", "边界", "boundary"]

        df["is_boundary"] = df["is_boundary"].apply(normalize_bool)

    return df


def build_kv_df(data: Dict[str, Any], key_name: str, value_name: str) -> pd.DataFrame:
    if not isinstance(data, dict) or not data:
        return pd.DataFrame(columns=[key_name, value_name])
    return pd.DataFrame(list(data.items()), columns=[key_name, value_name])


def pick_metric_columns(df: pd.DataFrame):
    """
    从临床联合表里识别真正的 ARM 功能指标。
    明确排除患者编号、聚类标签、置信度、版本信息等非生理指标。
    """
    preferred_keywords = [
        "resting_pressure",
        "msp",
        "squeeze_duration",
        "defecatory_rectal_pressure",
        "max_tolerable_volume",
        "anal_length",
        "rair_min_volume",
        "first_sensation",
        "desire_to_defecate",
        "urgency_threshold",

        "肛门括约肌静息压",
        "静息压",
        "最大缩榨压",
        "缩榨压",
        "缩肛持续时间",
        "排便时直肠压力",
        "最大容量感觉阈值",
        "最大耐受容量",
        "肛门括约肌长度",
        "肛管长度",
        "RAIR诱发最小容积",
        "初始感觉阈值",
        "初始便意阈值",
        "排便窘迫感阈值",
    ]

    exclude_keywords = [
        "pid",
        "patient",
        "id",
        "key",
        "编号",
        "病例号",
        "姓名",
        "name",
        "cluster",
        "label",
        "分型",
        "consensus",
        "confidence",
        "置信度",
        "boundary",
        "边界",
        "stable",
        "稳定",
        "version",
        "版本",
        "fold",
        "seed",
        "index",
    ]

    metric_cols = []

    for col in df.columns:
        col_str = str(col)
        col_lower = col_str.lower()

        if any(k in col_lower for k in exclude_keywords):
            continue

        if not any(k.lower() in col_lower for k in preferred_keywords):
            continue

        values = pd.to_numeric(df[col], errors="coerce")
        if values.notnull().sum() > 0:
            metric_cols.append(col)

    return metric_cols[:12]


def build_median_profile(cluster_df: pd.DataFrame, metric_cols) -> Dict[str, float]:
    profile = {}
    for col in metric_cols:
        values = pd.to_numeric(cluster_df[col], errors="coerce").dropna()
        if not values.empty:
            profile[col] = float(values.median())
    return profile


def build_boundary_summary(cluster_df: pd.DataFrame) -> Dict[str, Any]:
    """
    只汇总边界/稳定性信息。
    如果当前版本没有可靠的 is_boundary 字段，则不强行显示 100% 稳定。
    """
    if "is_boundary" not in cluster_df.columns:
        return {}

    s = cluster_df["is_boundary"].dropna()

    if s.empty:
        return {}

    s_bool = s.astype(bool)
    boundary_n = int(s_bool.sum())
    total_n = int(len(s_bool))
    boundary_rate = float(boundary_n / total_n) if total_n > 0 else None

    return {
        "可评估样本数": total_n,
        "边界患者数": boundary_n,
        "边界患者比例": boundary_rate,
        "稳定患者比例": 1.0 - boundary_rate if boundary_rate is not None else None,
    }


def build_data_quality_table(cluster_df: pd.DataFrame, metric_cols) -> pd.DataFrame:
    """
    数据质量表：展示核心功能指标的有效样本数、缺失数和缺失比例。
    不在这里伪造医学异常阈值。
    """
    rows = []
    total_n = len(cluster_df)

    if total_n == 0:
        return pd.DataFrame(columns=["指标", "有效样本数", "缺失数", "缺失比例"])

    for col in metric_cols:
        if col not in cluster_df.columns:
            continue

        values = pd.to_numeric(cluster_df[col], errors="coerce")
        valid_n = int(values.notna().sum())
        missing_n = int(total_n - valid_n)
        missing_rate = float(missing_n / total_n)

        rows.append(
            {
                "指标": col,
                "有效样本数": valid_n,
                "缺失数": missing_n,
                "缺失比例": missing_rate,
            }
        )

    return pd.DataFrame(rows)


def find_numeric_col_by_keywords(df: pd.DataFrame, keywords):
    """
    根据关键词模糊匹配真实列名，并确保该列可以转为数值。
    用于兼容 RAIR诱发最小容积(ml) 这类带单位的列名。
    """
    for keyword in keywords:
        keyword_lower = str(keyword).lower()

        for col in df.columns:
            col_lower = str(col).lower()

            if keyword_lower not in col_lower:
                continue

            values = pd.to_numeric(df[col], errors="coerce").dropna()
            if not values.empty:
                return col

    return None


def build_rair_stats(cluster_df: pd.DataFrame) -> pd.DataFrame:
    """
    RAIR 统计：按关键词模糊识别 RAIR 相关数值列。
    """
    candidate_map = {
        "RAIR诱发最小容积": [
            "rair_min_volume",
            "rair诱发最小容积",
            "rair最小容积",
            "rair_min",
        ],
        "松弛幅度": [
            "relaxation_amplitude",
            "松弛幅度",
            "松弛率",
        ],
        "最低点时间": [
            "t_min",
            "最低点时间",
            "最低点",
        ],
    }

    rows = []
    total_n = len(cluster_df)

    if total_n == 0:
        return pd.DataFrame()

    for display_name, keywords in candidate_map.items():
        col = find_numeric_col_by_keywords(cluster_df, keywords)

        if not col:
            continue

        values = pd.to_numeric(cluster_df[col], errors="coerce").dropna()

        if values.empty:
            continue

        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))

        rows.append(
            {
                "RAIR指标": display_name,
                "原始列名": col,
                "有效样本数": int(values.shape[0]),
                "有效比例": float(values.shape[0] / total_n),
                "中位数": float(values.median()),
                "四分位间距": f"{q1:.2f}–{q3:.2f}",
            }
        )

    return pd.DataFrame(rows)


def describe_cluster(cluster_id, size, stable_ratio, profile: Dict[str, Any]) -> str:
    if not profile:
        return (
            f"Cluster {cluster_id} 共包含 {size} 名患者，稳定患者比例为 {fmt_ratio(stable_ratio)}。"
            "当前版本暂未读取到可用于描述的核心临床指标。"
        )

    top_items = list(profile.items())[:4]
    metric_text = "；".join([f"{k}中位数为{fmt_number(v)}" for k, v in top_items])

    return (
        f"Cluster {cluster_id} 共包含 {size} 名患者，稳定患者比例为 {fmt_ratio(stable_ratio)}。"
        f"该集群的主要功能指标表现为：{metric_text}。"
        "该描述基于当前版本临床联合表的患者级统计结果，仅用于科研表型解释。"
    )


# ============================================================
# 页面标题
# ============================================================

st.title("🧩 集群功能表型视图")
st.caption(
    f"展示 AI 无监督分型得到的功能亚型整体生理特征，仅用于科研分析｜当前用户：{user.get('username', '-')}"
)

st.info(
    f"当前集群版本：**{current_version.get('display_name', selected_version)}**  \n"
    f"方法配置：**{current_version.get('method', '-')}**"
)

st.divider()


# ============================================================
# 读取当前版本 clinical_with_clusters
# ============================================================

clinical_path_raw = current_files.get("clinical_with_clusters")
clinical_path = resolve_path(clinical_path_raw)

boundary_threshold = current_version.get("boundary_threshold", 0.8)
try:
    boundary_threshold = float(boundary_threshold)
except Exception:
    boundary_threshold = 0.8

clinical_df_raw = safe_read_csv(clinical_path)
clinical_df = normalize_cluster_df(clinical_df_raw, boundary_threshold)

with st.expander("调试：查看当前版本集群文件路径"):
    st.write("clinical_with_clusters 原始路径：", clinical_path_raw)
    st.write("clinical_with_clusters 解析路径：", str(clinical_path) if clinical_path else None)
    st.write("文件是否存在：", clinical_path.exists() if clinical_path else False)
    st.write("是否读取成功：", clinical_df is not None and not clinical_df.empty)

    if clinical_df_raw is not None and not clinical_df_raw.empty:
        st.write("原始列名：", list(clinical_df_raw.columns))
    if clinical_df is not None and not clinical_df.empty:
        st.write("标准化后列名：", list(clinical_df.columns))

if clinical_df is None or clinical_df.empty:
    st.error("当前版本未读取到 clinical_with_clusters，请检查 versions.yaml 中 clinical_with_clusters 路径。")
    st.stop()


# ============================================================
# Sidebar 集群选择
# ============================================================

cluster_values = sorted(
    [
        x for x in clinical_df["consensus_cluster"].dropna().unique().tolist()
    ]
)

if not cluster_values:
    st.error("当前数据中没有 consensus_cluster / cluster 字段，无法展示集群视图。")
    st.stop()

st.sidebar.header("集群选择")
cluster_id = st.sidebar.selectbox(
    "选择集群（Cluster）",
    options=cluster_values,
    index=0,
)

# 分析人群切换
st.sidebar.header("分析人群")
population_cluster = st.sidebar.radio(
    "分析人群",
    options=["all", "stable"],
    index=0,
    format_func=lambda x: "全部患者" if x == "all" else "稳定患者 (conf>=0.8)",
    key="cluster_population",
)

# 应用人群过滤
if population_cluster == "stable":
    clinical_df = clinical_df[clinical_df["confidence"] >= boundary_threshold].copy()
    # 重新获取cluster values（稳定人群可能某个cluster为空）
    cluster_values_filtered = sorted(
        [x for x in clinical_df["consensus_cluster"].dropna().unique().tolist()]
    )
    if not cluster_values_filtered:
        st.error("稳定患者中未找到有效的集群数据。")
        st.stop()
    # 确保选中的cluster在过滤后存在
    if cluster_id not in cluster_values_filtered:
        cluster_id = cluster_values_filtered[0]

refresh = st.sidebar.button("刷新结果", key="cluster_refresh_btn")
if refresh:
    st.cache_data.clear()
    st.rerun()


# ============================================================
# 当前集群数据
# ============================================================

cluster_df = clinical_df[clinical_df["consensus_cluster"] == cluster_id].copy()

if cluster_df.empty:
    st.error("未找到该集群数据。")
    st.stop()

size = len(cluster_df)
if "is_boundary" in cluster_df.columns and cluster_df["is_boundary"].notna().any():
    stable_ratio = 1.0 - float(cluster_df["is_boundary"].dropna().mean())
else:
    stable_ratio = None

metric_cols = pick_metric_columns(clinical_df)
profile = build_median_profile(cluster_df, metric_cols)
boundary_summary = build_boundary_summary(cluster_df)
data_quality_df = build_data_quality_table(cluster_df, metric_cols)
rair_stats_df = build_rair_stats(cluster_df)
phenotype_description = describe_cluster(cluster_id, size, stable_ratio, profile)


# ============================================================
# 集群基本信息
# ============================================================

st.subheader("集群基本信息")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("集群编号", f"Cluster {cluster_id}")

with c2:
    st.metric("样本量", size)

with c3:
    st.metric("稳定患者比例", fmt_ratio(stable_ratio))

st.divider()


# ============================================================
# 核心功能指标画像
# ============================================================

st.subheader("核心功能指标画像（Core Functional Profile）")

df_profile = build_kv_df(profile, "指标", "中位数")

if df_profile.empty:
    st.info("暂无核心功能指标数据。")
else:
    st.bar_chart(df_profile.set_index("指标"))
    df_profile_show = df_profile.copy()
    df_profile_show["中位数"] = df_profile_show["中位数"].apply(lambda x: fmt_number(x, 2))
    st.dataframe(df_profile_show, use_container_width=True, hide_index=True)

st.divider()


# ============================================================
# 稳定性与数据完整性
# ============================================================

st.subheader("稳定性与数据完整性（Stability / Data Completeness）")

st.caption(
    "本模块仅展示边界患者比例和核心功能指标缺失情况；"
    "当前页面不根据医学参考范围自动判定“异常”。"
)

if boundary_summary:
    q1, q2, q3, q4 = st.columns(4)

    with q1:
        st.metric("可评估样本数", boundary_summary.get("可评估样本数", "-"))

    with q2:
        st.metric("边界患者数", boundary_summary.get("边界患者数", "-"))

    with q3:
        st.metric("边界患者比例", fmt_ratio(boundary_summary.get("边界患者比例")))

    with q4:
        st.metric("稳定患者比例", fmt_ratio(boundary_summary.get("稳定患者比例")))
else:
    st.info("当前版本未提供可靠的边界患者 / 稳定性字段，因此不显示稳定性比例。")

st.markdown("#### 核心功能指标数据完整性")

if data_quality_df.empty:
    st.info("暂无核心功能指标数据质量统计。")
else:
    dq_show = data_quality_df.copy()
    dq_show["缺失比例"] = dq_show["缺失比例"].apply(fmt_ratio)

    st.dataframe(
        dq_show,
        use_container_width=True,
        hide_index=True,
    )

    missing_nonzero = data_quality_df[data_quality_df["缺失比例"] > 0].copy()

    if missing_nonzero.empty:
        st.success("当前集群核心功能指标无缺失。")
    else:
        st.markdown("#### 存在缺失的指标")
        st.bar_chart(
            missing_nonzero.set_index("指标")["缺失比例"]
        )

st.divider()

# ============================================================
# RAIR 统计
# ============================================================

st.subheader("RAIR 相关统计（RAIR-related Statistics）")

if rair_stats_df is None or rair_stats_df.empty:
    st.info("当前版本未识别到可用的 RAIR 数值列。请在调试区查看原始列名。")
else:
    rair_show = rair_stats_df.copy()
    rair_show["有效比例"] = rair_show["有效比例"].apply(fmt_ratio)
    rair_show["中位数"] = rair_show["中位数"].apply(lambda x: fmt_number(x, 2))

    st.dataframe(
        rair_show,
        use_container_width=True,
        hide_index=True,
    )

st.divider()


# ============================================================
# 集群生理特征总结
# ============================================================

st.subheader("集群生理特征总结")
st.markdown(phenotype_description)

st.divider()


# ============================================================
# 当前集群患者列表
# ============================================================

st.subheader("当前集群患者列表")

show_cols = [
    c for c in [
        "patient_id",
        "consensus_cluster",
        "confidence",
        "is_boundary",
    ] + metric_cols[:6]
    if c in cluster_df.columns
]

cluster_show = cluster_df[show_cols].copy()

if "confidence" in cluster_show.columns:
    cluster_show["confidence"] = cluster_show["confidence"].apply(lambda x: fmt_number(x, 3))

if "is_boundary" in cluster_show.columns:
    cluster_show["is_boundary"] = cluster_show["is_boundary"].apply(lambda x: "是" if bool(x) else "否")

st.dataframe(cluster_show, use_container_width=True, hide_index=True)

st.divider()

st.caption("⚠️ 本页面展示的集群特征仅用于科研与功能表型分析，不用于临床诊断。")