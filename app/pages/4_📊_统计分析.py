import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st

from backend.auth.auth_service import require_role
from backend.version_manager import select_version_sidebar, safe_read_csv, resolve_path


st.set_page_config(page_title="统计分析 | ARM 功能表型系统", layout="wide")


# ============================================================
# 权限控制
# ============================================================

user = require_role("admin", "doctor")


# ============================================================
# 版本选择
# ============================================================

selected_version, current_version, current_files = select_version_sidebar(
    key="statistics_selected_version"
)


# ============================================================
# 工具函数
# ============================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    兼容不同统计 CSV 的列名。
    只在展示层做轻量统一，不改原始文件。
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    rename_map = {
        "metric": "指标",
        "label": "指标",
        "n": "样本量",
        "sample_size": "样本量",
        "H_value": "H",
        "H_statistic": "H",
        "statistic": "H",
        "p_value": "p_raw",
        "p": "p_raw",
        "p_adj": "p_adj_holm",
        "p_holm": "p_adj_holm",
        "holm_p": "p_adj_holm",
        "epsilon2": "epsilon_squared",
        "epsilon_square": "epsilon_squared",
        "effect_size": "epsilon_squared",
        "cluster_1": "cluster_i",
        "cluster_2": "cluster_j",
        "group1": "cluster_i",
        "group2": "cluster_j",
        "comparison_i": "cluster_i",
        "comparison_j": "cluster_j",
    }

    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    return df


def add_version_column(df: pd.DataFrame, version_short_name: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    if "版本" not in df.columns:
        df["版本"] = version_short_name

    if "分析人群" not in df.columns:
        df["分析人群"] = "stable"

    return df


def get_available_metrics(kruskal_df: pd.DataFrame, dunn_df: pd.DataFrame):
    metrics = []

    if kruskal_df is not None and not kruskal_df.empty and "指标" in kruskal_df.columns:
        metrics.extend(kruskal_df["指标"].dropna().astype(str).unique().tolist())

    if dunn_df is not None and not dunn_df.empty and "指标" in dunn_df.columns:
        metrics.extend(dunn_df["指标"].dropna().astype(str).unique().tolist())

    metrics = sorted(list(set(metrics)))
    return ["全部指标"] + metrics if metrics else ["全部指标"]


def filter_by_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if metric == "全部指标":
        return df

    if "指标" not in df.columns:
        return df

    return df[df["指标"].astype(str) == str(metric)]


def format_p_value(x):
    try:
        x = float(x)
        if x < 0.001:
            return "<0.001"
        return f"{x:.4f}"
    except Exception:
        return x


def format_float(x, digits=3):
    try:
        return round(float(x), digits)
    except Exception:
        return x


# ============================================================
# 页面标题
# ============================================================

st.title("📊 Kruskal-Wallis 与 Dunn 统计结果")
st.caption(
    f"展示不同共识分型之间的组间统计检验结果，仅用于科研分析｜当前用户：{user.get('username', '-')}"
)

st.info(
    f"当前统计版本：**{current_version.get('display_name', selected_version)}**  \n"
    f"方法配置：**{current_version.get('method', '-')}**"
)


# ============================================================
# 读取当前版本统计文件
# ============================================================

kruskal_path_raw = current_files.get("kruskal_stable")
dunn_path_raw = current_files.get("dunn_stable")

kruskal_path = resolve_path(kruskal_path_raw)
dunn_path = resolve_path(dunn_path_raw)

kruskal_df = safe_read_csv(kruskal_path)
dunn_df = safe_read_csv(dunn_path)

version_short_name = current_version.get("short_name", selected_version)

kruskal_df = normalize_columns(kruskal_df)
dunn_df = normalize_columns(dunn_df)

kruskal_df = add_version_column(kruskal_df, version_short_name)
dunn_df = add_version_column(dunn_df, version_short_name)


with st.expander("调试：查看当前版本统计文件路径"):
    st.write("Kruskal 原始路径：", kruskal_path_raw)
    st.write("Kruskal 解析路径：", str(kruskal_path) if kruskal_path else None)
    st.write("Kruskal 文件是否存在：", kruskal_path.exists() if kruskal_path else False)
    st.write("Kruskal 是否读取成功：", kruskal_df is not None and not kruskal_df.empty)

    st.write("Dunn 原始路径：", dunn_path_raw)
    st.write("Dunn 解析路径：", str(dunn_path) if dunn_path else None)
    st.write("Dunn 文件是否存在：", dunn_path.exists() if dunn_path else False)
    st.write("Dunn 是否读取成功：", dunn_df is not None and not dunn_df.empty)

    if kruskal_df is not None and not kruskal_df.empty:
        st.write("Kruskal 列名：", list(kruskal_df.columns))

    if dunn_df is not None and not dunn_df.empty:
        st.write("Dunn 列名：", list(dunn_df.columns))


if (kruskal_df is None or kruskal_df.empty) and (dunn_df is None or dunn_df.empty):
    st.error("当前版本未读取到 Kruskal 或 Dunn 统计结果，请检查 versions.yaml 中的统计文件路径。")
    st.stop()


# ============================================================
# 指标选择
# ============================================================

available_metrics = get_available_metrics(kruskal_df, dunn_df)
metric = st.selectbox("选择指标", available_metrics, index=0)

kruskal_show = filter_by_metric(kruskal_df, metric)
dunn_show = filter_by_metric(dunn_df, metric)


# ============================================================
# 分析人群选择
# ============================================================

population_options = []
if kruskal_df is not None and not kruskal_df.empty and "分析人群" in kruskal_df.columns:
    population_options = sorted(kruskal_df["分析人群"].dropna().unique())
if not population_options:
    population_options = ["stable"]

population = st.selectbox(
    "分析人群",
    population_options,
    index=0,
    format_func=lambda x: "全部患者" if x == "all" else "稳定患者 (confidence≥0.8)",
)
if population == "stable" and version_short_name == "M1":
    st.caption("M1 全部患者置信度=1.0，两部分结果相同")

kruskal_show = kruskal_show[kruskal_show["分析人群"] == population] if kruskal_show is not None and not kruskal_show.empty and "分析人群" in kruskal_show.columns else kruskal_show
dunn_show = dunn_show[dunn_show["分析人群"] == population] if dunn_show is not None and not dunn_show.empty and "分析人群" in dunn_show.columns else dunn_show


# ============================================================
# 统计摘要
# ============================================================

st.subheader("统计摘要")

if metric == "全部指标":
    n_kruskal = len(kruskal_show) if kruskal_show is not None else 0
    n_dunn = len(dunn_show) if dunn_show is not None else 0

    sig_count = 0
    if kruskal_show is not None and not kruskal_show.empty:
        if "是否显著_adj" in kruskal_show.columns:
            sig_count = int(kruskal_show["是否显著_adj"].astype(str).isin(["True", "true", "1", "是", "显著"]).sum())
        elif "p_adj_holm" in kruskal_show.columns:
            sig_count = int((pd.to_numeric(kruskal_show["p_adj_holm"], errors="coerce") < 0.05).sum())

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Kruskal 主检验条数", n_kruskal)
    with c2:
        st.metric("Dunn 事后检验条数", n_dunn)
    with c3:
        st.metric("显著主检验指标数", sig_count)

    st.info(
        f"当前展示 {version_short_name} 的统计结果。"
        f"若切换 M1–M5，系统会自动读取对应版本的统计 CSV。"
    )
else:
    st.info(f"当前展示指标：**{metric}**")


# ============================================================
# Kruskal-Wallis 主检验
# ============================================================

st.subheader("Kruskal-Wallis 主检验")

if kruskal_show is None or kruskal_show.empty:
    st.warning("暂无 Kruskal-Wallis 主检验结果。")
else:
    kruskal_display = kruskal_show.copy()

    for col in ["p_raw", "p_adj_holm"]:
        if col in kruskal_display.columns:
            kruskal_display[col] = kruskal_display[col].apply(format_p_value)

    for col in ["H", "epsilon_squared"]:
        if col in kruskal_display.columns:
            kruskal_display[col] = kruskal_display[col].apply(lambda x: format_float(x, 4))

    show_cols = [
        c for c in [
            "指标",
            "样本量",
            "H",
            "p_raw",
            "p_adj_holm",
            "显著性_adj",
            "是否显著_adj",
            "epsilon_squared",
            "effect_size_label",
            "分析人群",
            "版本",
        ] if c in kruskal_display.columns
    ]

    if show_cols:
        st.dataframe(kruskal_display[show_cols], use_container_width=True, hide_index=True)
    else:
        st.dataframe(kruskal_display, use_container_width=True, hide_index=True)


# ============================================================
# Dunn 事后检验
# ============================================================

st.subheader("Dunn 事后检验")

if dunn_show is None or dunn_show.empty:
    st.warning("暂无 Dunn 事后检验结果。")
else:
    dunn_display = dunn_show.copy()

    for col in ["p_adj", "p_raw", "p_adj_holm"]:
        if col in dunn_display.columns:
            dunn_display[col] = dunn_display[col].apply(format_p_value)

    show_cols = [
        c for c in [
            "指标",
            "cluster_i",
            "cluster_j",
            "p_adj",
            "p_raw",
            "p_adj_holm",
            "significance",
            "显著性",
            "分析人群",
            "版本",
        ] if c in dunn_display.columns
    ]

    if show_cols:
        st.dataframe(dunn_display[show_cols], use_container_width=True, hide_index=True)
    else:
        st.dataframe(dunn_display, use_container_width=True, hide_index=True)


st.divider()
st.caption("⚠️ 本页面用于科研统计分析，不用于临床诊断或治疗决策。")