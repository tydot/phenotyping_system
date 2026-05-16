import sys
from pathlib import Path

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


st.set_page_config(page_title="Stability View | ARM 功能表型系统", layout="wide")


# ============================================================
# 权限控制
# ============================================================

user = require_role("admin", "doctor")


# ============================================================
# 版本选择
# ============================================================

selected_version, current_version, current_files = select_version_sidebar(
    key="stability_selected_version"
)


# ============================================================
# 工具函数
# ============================================================

def fmt_pct(x, digits=1):
    try:
        return f"{float(x) * 100:.{digits}f}%"
    except Exception:
        return "-"


def fmt_num(x, digits=3):
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "-"


def find_first_existing_col(df: pd.DataFrame, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalize_stability_df(df: pd.DataFrame, boundary_threshold: float) -> pd.DataFrame:
    """
    兼容不同 consensus_labels.csv 的列名。
    统一生成：
    - patient_id
    - consensus_cluster
    - confidence
    - is_boundary
    - stability_label
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

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
        df["is_boundary"] = df["confidence"].apply(
            lambda x: bool(pd.notnull(x) and float(x) < boundary_threshold)
        )
    else:
        def normalize_bool(x):
            if isinstance(x, bool):
                return x
            s = str(x).strip().lower()
            return s in ["true", "1", "yes", "y", "是", "边界", "boundary"]

        df["is_boundary"] = df["is_boundary"].apply(normalize_bool)

    df["stability_label"] = df["is_boundary"].apply(
        lambda x: "边界患者" if bool(x) else "稳定患者"
    )

    if "switch_rate" not in df.columns:
        if "confidence" in df.columns:
            df["switch_rate"] = df["confidence"].apply(
                lambda x: 1 - float(x) if pd.notnull(x) else None
            )
        else:
            df["switch_rate"] = None

    return df


def build_cluster_stability(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "consensus_cluster" not in df.columns:
        return pd.DataFrame()

    rows = []

    for cluster_id, group in df.groupby("consensus_cluster", dropna=False):
        n = len(group)
        n_boundary = int(group["is_boundary"].sum()) if "is_boundary" in group.columns else 0
        n_stable = n - n_boundary
        stable_ratio = n_stable / n if n else 0

        rows.append(
            {
                "Cluster": f"Cluster {cluster_id}",
                "患者数": n,
                "稳定患者数": n_stable,
                "边界患者数": n_boundary,
                "稳定比例": stable_ratio,
            }
        )

    return pd.DataFrame(rows)


def build_confidence_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "confidence" not in df.columns:
        return pd.DataFrame()

    conf = pd.to_numeric(df["confidence"], errors="coerce").dropna()

    if conf.empty:
        return pd.DataFrame()

    bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.000001]
    labels = ["<0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]

    dist = pd.cut(conf, bins=bins, labels=labels, include_lowest=True)
    result = dist.value_counts().sort_index().reset_index()
    result.columns = ["confidence区间", "患者数"]

    return result


# ============================================================
# 页面标题
# ============================================================

st.title("📊 Stability View")
st.caption(
    f"基于 patient consensus 结果的群体稳定性与边界患者分析｜当前用户：{user.get('username', '-')}"
)

st.info(
    f"当前稳定性版本：**{current_version.get('display_name', selected_version)}**  \n"
    f"方法配置：**{current_version.get('method', '-')}**"
)

st.divider()


# ============================================================
# 读取当前版本 consensus_labels.csv
# ============================================================

consensus_path_raw = current_files.get("consensus_labels")
consensus_path = resolve_path(consensus_path_raw)

boundary_threshold = current_version.get("boundary_threshold", 0.8)
try:
    boundary_threshold = float(boundary_threshold)
except Exception:
    boundary_threshold = 0.8

if st.button("刷新稳定性结果"):
    st.cache_data.clear()
    st.rerun()

consensus_df_raw = safe_read_csv(consensus_path)
consensus_df = normalize_stability_df(consensus_df_raw, boundary_threshold)

with st.expander("调试：查看当前版本稳定性文件路径"):
    st.write("consensus_labels 原始路径：", consensus_path_raw)
    st.write("consensus_labels 解析路径：", str(consensus_path) if consensus_path else None)
    st.write("文件是否存在：", consensus_path.exists() if consensus_path else False)
    st.write("是否读取成功：", consensus_df is not None and not consensus_df.empty)

    if consensus_df_raw is not None and not consensus_df_raw.empty:
        st.write("原始列名：", list(consensus_df_raw.columns))
    if consensus_df is not None and not consensus_df.empty:
        st.write("标准化后列名：", list(consensus_df.columns))

if consensus_df is None or consensus_df.empty:
    st.error("当前版本未读取到 consensus_labels.csv，请检查 versions.yaml 中 consensus_labels 路径。")
    st.stop()


# ============================================================
# 整体稳定性概览
# ============================================================

n_total = len(consensus_df)
n_boundary = int(consensus_df["is_boundary"].sum()) if "is_boundary" in consensus_df.columns else 0
n_stable = n_total - n_boundary

stable_ratio = n_stable / n_total if n_total else 0
boundary_ratio = n_boundary / n_total if n_total else 0

st.subheader("整体稳定性概览")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("患者总数", n_total)

with c2:
    st.metric("稳定患者数", n_stable)

with c3:
    st.metric("边界患者数", n_boundary)

with c4:
    st.metric("稳定患者比例", fmt_pct(stable_ratio))

st.caption(
    f"当前系统采用 confidence ≥ {boundary_threshold} 视为稳定患者，"
    f"confidence < {boundary_threshold} 视为边界患者。"
)

st.divider()


# ============================================================
# 各 Cluster 稳定性 + Confidence 分布
# ============================================================

left, right = st.columns(2)

with left:
    st.subheader("各 Cluster 稳定性")

    cluster_stability_df = build_cluster_stability(consensus_df)

    if cluster_stability_df.empty:
        st.caption("暂无各 cluster 稳定性结果。")
    else:
        chart_df = cluster_stability_df[["Cluster", "稳定比例"]].copy()
        st.bar_chart(chart_df.set_index("Cluster"))

        show_df = cluster_stability_df.copy()
        show_df["稳定比例"] = show_df["稳定比例"].apply(fmt_pct)

        st.dataframe(show_df, use_container_width=True, hide_index=True)

with right:
    st.subheader("Confidence 分布")

    conf_dist_df = build_confidence_distribution(consensus_df)

    if conf_dist_df.empty:
        st.caption("暂无 confidence 分布结果。")
    else:
        st.bar_chart(conf_dist_df.set_index("confidence区间"))
        st.dataframe(conf_dist_df, use_container_width=True, hide_index=True)

st.divider()



# ============================================================
# 边界患者列表
# ============================================================

st.subheader("边界患者列表")

boundary_df = consensus_df[consensus_df["is_boundary"] == True].copy()  # noqa: E712

if boundary_df.empty:
    st.success("当前版本没有边界患者，或所有患者均达到稳定阈值。")
else:
    show_cols = [
        c for c in [
            "patient_id",
            "consensus_cluster",
            "confidence",
            "switch_rate",
            "stability_label",
        ] if c in boundary_df.columns
    ]

    boundary_show = boundary_df[show_cols].copy()

    if "confidence" in boundary_show.columns:
        boundary_show["confidence"] = boundary_show["confidence"].apply(lambda x: fmt_num(x, 3))

    if "switch_rate" in boundary_show.columns:
        boundary_show["switch_rate"] = boundary_show["switch_rate"].apply(lambda x: fmt_num(x, 3))

    st.dataframe(boundary_show, use_container_width=True, hide_index=True)

st.divider()



# ============================================================
# AI 解释
# ============================================================

st.subheader("AI 解释")

st.markdown("**总体评估**")
if n_total == 0:
    st.write("暂无可分析患者。")
elif n_boundary == 0:
    st.write(
        f"当前版本 {current_version.get('short_name', selected_version)} 下，"
        "全部患者均达到稳定阈值，说明多随机种子聚类结果具有较高一致性。"
    )
else:
    st.write(
        f"当前版本 {current_version.get('short_name', selected_version)} 下，"
        f"稳定患者比例为 {fmt_pct(stable_ratio)}，边界患者比例为 {fmt_pct(boundary_ratio)}。"
        "边界患者提示其处于不同功能亚型之间的过渡区域，后续解释时应结合具体临床指标。"
    )

st.markdown("**Cluster 分析**")
if not cluster_stability_df.empty:
    best_row = cluster_stability_df.sort_values("稳定比例", ascending=False).iloc[0]
    weak_row = cluster_stability_df.sort_values("稳定比例", ascending=True).iloc[0]

    st.write(
        f"{best_row['Cluster']} 的稳定比例最高，为 {fmt_pct(best_row['稳定比例'])}；"
        f"{weak_row['Cluster']} 的稳定比例最低，为 {fmt_pct(weak_row['稳定比例'])}。"
    )
else:
    st.write("暂无 Cluster 稳定性分析。")

st.markdown("**Confidence 分析**")
if "confidence" in consensus_df.columns:
    conf_series = pd.to_numeric(consensus_df["confidence"], errors="coerce").dropna()
    if not conf_series.empty:
        st.write(
            f"当前版本 confidence 中位数为 {fmt_num(conf_series.median(), 3)}，"
            f"均值为 {fmt_num(conf_series.mean(), 3)}。"
        )
    else:
        st.write("暂无有效 confidence 数值。")
else:
    st.write("当前 consensus 文件未提供 confidence 字段。")

st.markdown("**建议**")
st.write(
    "- 稳定患者可作为主要统计验证对象。  \n"
    "- 边界患者建议在患者页中保留分型置信度提示。  \n"
    "- 比较 M1–M5 时，应同时关注稳定患者比例、边界患者比例和临床指标显著性。"
)

st.divider()

with st.expander("调试：查看标准化后的 Stability 数据"):
    st.dataframe(consensus_df, use_container_width=True, hide_index=True)

st.caption("⚠️ Stability View 反映的是多随机种子下聚类一致性，不等同于临床诊断确定性。")