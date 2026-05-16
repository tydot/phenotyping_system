# app/app.py
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import yaml
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="ARM功能表型系统",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# 版本配置读取
# =========================
VERSION_CONFIG_PATH = ROOT_DIR / "config" / "versions.yaml"


@st.cache_data
def load_versions_config():
    """
    读取 M1-M5 分型版本配置文件。
    配置文件位置：config/versions.yaml
    """
    with open(VERSION_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_current_version_config():
    """
    在 sidebar 中提供版本选择，并返回当前版本配置。
    """
    if not VERSION_CONFIG_PATH.exists():
        st.sidebar.error(f"未找到版本配置文件：{VERSION_CONFIG_PATH}")
        st.stop()

    version_config = load_versions_config()

    versions = version_config.get("versions", {})
    if not versions:
        st.sidebar.error("versions.yaml 中未找到 versions 配置。")
        st.stop()

    version_order = version_config.get("version_order", list(versions.keys()))
    default_version = version_config.get("default_version", version_order[0])

    if default_version not in version_order:
        default_version = version_order[0]

    selected_version = st.sidebar.selectbox(
        "选择分型版本",
        version_order,
        index=version_order.index(default_version),
        format_func=lambda key: versions[key].get("display_name", key)
    )

    current_version = versions[selected_version]

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**当前版本：** {current_version.get('short_name', selected_version)}")
    st.sidebar.markdown(f"**当前方法：** {current_version.get('method', '-')}")
    st.sidebar.caption(current_version.get("description", ""))

    return selected_version, current_version


selected_version_key, current_version = get_current_version_config()
current_files = current_version.get("files", {})

# =========================
# 基础路径
# =========================
umap_file = current_files.get("umap_figure")
UMAP_PATH = Path(umap_file) if umap_file else None

# =========================
# 首页标题
# =========================
st.title("基于直肠测压图像的排便障碍患者亚型分类系统")
st.caption("AI 驱动的功能表型分析科研原型系统")

st.info(
    "本系统面向排便障碍患者功能表型研究，"
    "结合图像侧无监督分型、共识分型、临床指标分析及可视化展示，"
    "实现患者级、集群级和总体队列级的辅助分析。\n\n"
    "⚠️ 本系统仅用于科研与功能表型分析，不用于临床诊断或治疗决策。"
)

st.divider()

# =========================
# 当前版本说明
# =========================
st.subheader("当前分型版本")

v1, v2, v3, v4 = st.columns(4)

with v1:
    st.metric("版本", current_version.get("short_name", selected_version_key))

with v2:
    role = current_version.get("role", "-")
    role_display = "主实验版本" if role == "main" else "消融对照版本"
    st.metric("版本角色", role_display)

with v3:
    st.metric("聚类数量", f"{current_version.get('n_clusters', 3)} 类")

with v4:
    st.metric("边界阈值", f"confidence < {current_version.get('boundary_threshold', 0.8)}")

st.markdown(
    f"""
**方法配置：** {current_version.get("method", "-")}  

**版本说明：** {current_version.get("description", "-")}
"""
)

st.divider()

# =========================
# 系统目标
# =========================
st.subheader("系统目标")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 患者亚型发现")
    st.write(
        "基于多协议 ARM / RAIR 图像构建患者级表征，"
        "通过无监督聚类探索潜在功能亚型。"
    )

with col2:
    st.markdown("### 临床意义验证")
    st.write(
        "将共识分型结果与测压指标、RAIR 特征及 Rome IV proxy 框架结合，"
        "验证分型的病理生理相关性。"
    )

with col3:
    st.markdown("### 辅助解释与展示")
    st.write(
        "通过患者视图、集群视图、稳定性分析和总体队列视图，"
        "为科研分析提供可视化支持。"
    )

st.divider()

# =========================
# 系统流程
# =========================
st.subheader("系统流程")

st.markdown(
    """
**数据输入**  
ARM 图像 / RAIR 图像 / 临床报告

**患者级表征构建**  
图像特征提取与协议级聚合

**共识分型**  
多随机种子聚类、标签对齐、投票生成最终亚型

**临床分析**  
组间分析、组内分析、Rome IV proxy 对照、RAIR 验证

**结果展示**  
总体队列视图 / 患者视图 / 集群视图 / 稳定性分析
"""
)

st.divider()

# =========================
# 页面入口说明
# =========================
st.subheader("功能模块")

c1, c2 = st.columns(2)

with c1:
    st.markdown("### 📊 总体队列视图")
    st.write("查看研究队列规模、协议覆盖率、帧数分布及数据质量概况。")

    st.markdown("### 🧠 患者视图")
    st.write("查看单个患者的 AI 分型结果、稳定性置信度及关键 ARM 临床指标。")

with c2:
    st.markdown("### 🧩 集群视图")
    st.write("查看不同功能亚型的样本量、稳定比例、功能画像及异常比例。")

    st.markdown("### 🔒 稳定性分析")
    st.write("查看多随机种子条件下的共识分型稳定性、边界患者比例与置信度分布。")

st.divider()

# =========================
# 关键结果摘要
# =========================
st.subheader("关键结果摘要")

r1, r2, r3, r4 = st.columns(4)

with r1:
    st.metric("功能亚型数量", f"{current_version.get('n_clusters', 3)} 类")

with r2:
    st.metric("主结果来源", current_version.get("short_name", "M1"))

with r3:
    st.metric("边界判定阈值", f"confidence < {current_version.get('boundary_threshold', 0.8)}")

with r4:
    st.metric("当前阶段", "科研原型")

st.markdown(
    f"""
- 系统基于无监督聚类识别出 **{current_version.get('n_clusters', 3)} 个具有明确生理差异的功能亚型**。  
- 当前展示版本为 **{current_version.get('display_name', selected_version_key)}**。  
- 当前方法配置为 **{current_version.get('method', '-')}**。  
- 共识分型结果与临床联合表已接入，可展示关键 ARM 功能指标。  
- 目前系统重点用于 **功能表型研究与方法验证**，后续将继续接入 Rome IV proxy、RAIR 特征及专家知识增强模块。  
"""
)

st.divider()

# =========================
# UMAP 总览图
# =========================
st.subheader("分型结构总览")

if UMAP_PATH is None:
    st.warning("当前版本配置中未设置 umap_figure 路径。")
elif UMAP_PATH.exists():
    try:
        img = Image.open(UMAP_PATH)
        st.image(
            img,
            caption=f"{current_version.get('short_name', selected_version_key)} 患者级 UMAP 共识分型结果",
            use_column_width=True
        )
    except Exception as e:
        st.error(f"图片读取失败: {e}")
else:
    st.error(f"未找到图片: {UMAP_PATH}")

st.divider()

# =========================
# 使用说明
# =========================
st.subheader("使用说明")

st.markdown(
    """
1. 使用左侧导航栏切换到不同功能页面。  
2. 在首页左侧选择 M1-M5 分型版本，可查看不同算法配置下的 UMAP 分型结构。  
3. 在**患者视图**中输入患者编号，可查看其 AI 分型与生理证据。  
4. 在**集群视图**中查看不同 cluster 的功能画像与异常比例。  
5. 在**稳定性分析**中查看稳定患者与边界患者分布。  
"""
)

st.caption(
    "科研声明：本系统仅用于排便障碍功能表型研究与方法验证，不用于临床诊断或治疗决策。"
)