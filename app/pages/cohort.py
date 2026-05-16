import sys
from pathlib import Path
from PIL import Image
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st

from backend.api.cohort import get_cohort_view
from backend.auth.auth_service import require_role
from backend.version_manager import select_version_sidebar, resolve_path


st.set_page_config(page_title="Cohort Overview", layout="wide")


# ============================================================
# 权限控制
# ============================================================

# 只有管理员和医生可访问
user = require_role("admin", "doctor")


# ============================================================
# 版本选择
# ============================================================

selected_version, current_version, current_files = select_version_sidebar(
    key="cohort_selected_version"
)


# ============================================================
# 数据加载
# ============================================================

@st.cache_data(show_spinner=False)
def load_cohort_view():
    return get_cohort_view()


# ============================================================
# 页面标题
# ============================================================

st.title("📦 总体队列视图")
st.caption(
    f"基于 patient_clinical 临床联合表的 cohort 总览｜当前用户：{user.get('username', '-')}"
)

st.info(
    f"当前分型版本：**{current_version.get('display_name', selected_version)}**  \n"
    f"方法配置：**{current_version.get('method', '-')}**"
)

if st.button("刷新队列结果"):
    load_cohort_view.clear()
    st.rerun()


# ============================================================
# Cohort 数据
# ============================================================

data = load_cohort_view() or {}
overview = data.get("overview", {})
field_coverage = data.get("clinical_field_coverage", [])
clinical_summary = data.get("clinical_summary", [])
summary_text = data.get("summary_text", "暂无系统摘要。")


# ============================================================
# 队列概览
# ============================================================

st.subheader("队列概览")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("患者总数", overview.get("n_patients", 0))

with c2:
    st.metric("稳定患者数", overview.get("n_stable", 0))

with c3:
    st.metric("边界患者数", overview.get("n_boundary", 0))

with c4:
    stable_ratio = overview.get("stable_ratio")
    try:
        stable_ratio = f"{float(stable_ratio):.1%}"
    except Exception:
        stable_ratio = "-"
    st.metric("稳定患者比例", stable_ratio)

st.divider()


# ============================================================
# Cluster 分布
# ============================================================

st.subheader("Cluster 分布")

cluster_df = pd.DataFrame(overview.get("cluster_dist", []))

if cluster_df.empty:
    st.warning("暂无 cluster 分布数据。")
else:
    cluster_df = cluster_df.rename(
        columns={
            "consensus_cluster": "Cluster",
            "n": "患者数",
        }
    )

    st.dataframe(cluster_df, use_container_width=True, hide_index=True)

    if "Cluster" in cluster_df.columns and "患者数" in cluster_df.columns:
        st.bar_chart(cluster_df.set_index("Cluster"))

st.divider()


# ============================================================
# UMAP 图片展示：随 M1-M5 版本切换
# ============================================================

st.subheader("患者级 UMAP 可视化")

raw_umap_path = current_files.get("umap_figure")
IMG_PATH = resolve_path(raw_umap_path)

st.caption(
    f"当前版本：{current_version.get('display_name', selected_version)}｜"
    f"方法：{current_version.get('method', '-')}"
)

with st.expander("调试：查看 UMAP 图片路径"):
    st.write("versions.yaml 原始路径：", raw_umap_path)
    st.write("解析后的路径：", str(IMG_PATH) if IMG_PATH else None)
    st.write("图片是否存在：", IMG_PATH.exists() if IMG_PATH else False)

if IMG_PATH and IMG_PATH.exists():
    try:
        img = Image.open(IMG_PATH)
        st.image(
            img,
            caption=f"{current_version.get('short_name', selected_version)} 患者级 UMAP 共识分型结果",
            use_column_width=True,
        )
    except Exception as e:
        st.error(f"图片读取失败：{e}")
else:
    st.warning(f"未找到当前版本的 UMAP 图片文件：{IMG_PATH}")


# ============================================================
# 临床字段完整度
# ============================================================

st.subheader("临床字段完整度")

coverage_df = pd.DataFrame(field_coverage)

if coverage_df.empty:
    st.warning("暂无字段完整度数据。")
else:
    coverage_df_show = coverage_df.copy()

    if "coverage_rate" in coverage_df_show.columns:
        coverage_df_show["coverage_rate"] = coverage_df_show["coverage_rate"].map(
            lambda x: f"{x:.1%}" if pd.notnull(x) else "-"
        )

    coverage_df_show = coverage_df_show.rename(
        columns={
            "label": "字段名称",
            "field": "字段编码",
            "n_valid": "非空例数",
            "coverage_rate": "覆盖率",
        }
    )

    st.dataframe(coverage_df_show, use_container_width=True, hide_index=True)

st.divider()


# ============================================================
# 关键临床指标概览
# ============================================================

st.subheader("关键临床指标概览")

summary_df = pd.DataFrame(clinical_summary)

if summary_df.empty:
    st.warning("暂无关键指标概览数据。")
else:
    summary_df_show = summary_df.copy()

    for col in ["mean_value", "min_value", "max_value"]:
        if col in summary_df_show.columns:
            summary_df_show[col] = summary_df_show[col].apply(
                lambda x: round(x, 2) if pd.notnull(x) else None
            )

    summary_df_show = summary_df_show.rename(
        columns={
            "label": "指标名称",
            "field": "字段编码",
            "n_valid": "非空例数",
            "mean_value": "均值",
            "min_value": "最小值",
            "max_value": "最大值",
        }
    )

    st.dataframe(summary_df_show, use_container_width=True, hide_index=True)

st.divider()


# ============================================================
# 系统摘要
# ============================================================

st.subheader("系统摘要")
st.info(summary_text)

with st.expander("调试：查看原始 Cohort 输出"):
    st.json(data)

st.caption("⚠️ 本页面用于总体队列科研分析，不用于临床诊断或治疗决策。")