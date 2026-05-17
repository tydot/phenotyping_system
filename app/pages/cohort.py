import sys
from pathlib import Path
from PIL import Image

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


def find_first_existing_col(df: pd.DataFrame, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalize_patient_id(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def load_current_version_clinical(current_files):
    """
    cohort / patient 页面统一读取当前版本患者联合表。
    优先级：
    patient_clinical → cohort_table → merged_clinical → clinical_with_clusters
    """
    clinical_path_raw = (
        current_files.get("patient_clinical")
        or current_files.get("cohort_table")
        or current_files.get("merged_clinical")
        or current_files.get("clinical_with_clusters")
    )

    clinical_path = resolve_path(clinical_path_raw)
    clinical_df = safe_read_csv(clinical_path)

    return clinical_path_raw, clinical_path, clinical_df


def normalize_cohort_df(df: pd.DataFrame) -> pd.DataFrame:
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
            "PatientID",
            "Patient_ID",
            "病例号",
            "患者编号",
            "患者ID",
            "id",
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

    df["patient_id"] = df["patient_id"].apply(normalize_patient_id)

    if "consensus_cluster" not in df.columns:
        df["consensus_cluster"] = pd.NA

    if "confidence" in df.columns:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    else:
        df["confidence"] = pd.NA

    if "is_boundary" in df.columns:
        def normalize_bool(x):
            if pd.isna(x):
                return pd.NA
            if isinstance(x, bool):
                return x
            s = str(x).strip().lower()
            if s in ["true", "1", "yes", "y", "是", "边界", "boundary"]:
                return True
            if s in ["false", "0", "no", "n", "否", "稳定", "stable"]:
                return False
            return pd.NA

        df["is_boundary"] = df["is_boundary"].apply(normalize_bool)
    else:
        if "confidence" in df.columns and df["confidence"].notna().any():
            df["is_boundary"] = df["confidence"].apply(
                lambda x: bool(pd.notna(x) and float(x) < 0.8)
            )
        else:
            df["is_boundary"] = pd.NA

    return df

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

clinical_path_raw, clinical_path, clinical_df_raw = load_current_version_clinical(current_files)
cohort_df = normalize_cohort_df(clinical_df_raw)

with st.expander("调试：查看 cohort 数据路径"):
    st.write("clinical 原始路径：", clinical_path_raw)
    st.write("clinical 解析路径：", str(clinical_path) if clinical_path else None)
    st.write("文件是否存在：", clinical_path.exists() if clinical_path else False)
    st.write("是否读取成功：", cohort_df is not None and not cohort_df.empty)

    if clinical_df_raw is not None and not clinical_df_raw.empty:
        st.write("原始列名：", list(clinical_df_raw.columns))
        st.write("前 5 行：")
        st.dataframe(clinical_df_raw.head(), use_container_width=True)

    if cohort_df is not None and not cohort_df.empty:
        st.write("标准化后列名：", list(cohort_df.columns))
        st.write("前 20 个患者 ID：", cohort_df["patient_id"].head(20).tolist())

if cohort_df is None or cohort_df.empty:
    st.error("当前版本未读取到患者联合表。请检查 versions.yaml 中 patient_clinical / cohort_table / merged_clinical / clinical_with_clusters 路径。")
    st.stop()

if st.button("刷新队列结果"):
    st.cache_data.clear()
    st.rerun()


# ============================================================
# 队列概览
# ============================================================

total_n = int(cohort_df["patient_id"].nunique())

if "is_boundary" in cohort_df.columns and cohort_df["is_boundary"].notna().any():
    boundary_n = int(cohort_df["is_boundary"].fillna(False).astype(bool).sum())
    stable_n = int(total_n - boundary_n)
    stable_ratio = stable_n / total_n if total_n > 0 else 0
else:
    boundary_n = 0
    stable_n = 0
    stable_ratio = None

st.subheader("队列概览")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("患者总数", total_n)

with c2:
    st.metric("稳定患者数", stable_n if stable_ratio is not None else "-")

with c3:
    st.metric("边界患者数", boundary_n if stable_ratio is not None else "-")

with c4:
    st.metric("稳定患者比例", f"{stable_ratio:.1%}" if stable_ratio is not None else "-")

st.divider()


# ============================================================
# Cluster 分布
# ============================================================

st.subheader("Cluster 分布")

if "consensus_cluster" not in cohort_df.columns or cohort_df["consensus_cluster"].dropna().empty:
    st.warning("暂无 cluster 分布数据。")
else:
    cluster_count = (
        cohort_df["consensus_cluster"]
        .dropna()
        .value_counts()
        .sort_index()
        .rename_axis("Cluster")
        .reset_index(name="患者数")
    )

    st.bar_chart(cluster_count.set_index("Cluster"))
    st.dataframe(cluster_count, use_container_width=True, hide_index=True)

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


st.caption("⚠️ 本页面用于总体队列科研分析，不用于临床诊断或治疗决策。")