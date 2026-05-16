# backend/version_manager.py
"""
M1-M5 分型版本配置管理工具

用途：
1. 统一读取 config/versions.yaml
2. 为 Streamlit 页面提供版本选择控件
3. 获取当前版本对应的结果文件路径
4. 提供安全读取 CSV / 图片路径检查函数
5. 保持旧代码兼容：如果未传 version_key，则默认使用 default_version

说明：
本文件只负责“版本配置管理”，不负责算法训练、聚类或统计计算。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import yaml


# ============================================================
# 项目根目录与配置文件路径
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "versions.yaml"


# ============================================================
# 基础配置读取
# ============================================================

@st.cache_data(show_spinner=False)
def load_versions_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """
    读取版本配置文件。

    参数：
        config_path: versions.yaml 路径。默认读取 config/versions.yaml。

    返回：
        dict: YAML 配置内容。

    异常：
        FileNotFoundError: 配置文件不存在。
        ValueError: YAML 内容为空或格式不符合预期。
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"未找到版本配置文件：{path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("versions.yaml 内容为空或不是合法的字典结构。")

    if "versions" not in config or not isinstance(config["versions"], dict):
        raise ValueError("versions.yaml 中缺少 versions 字段，或 versions 不是字典结构。")

    if not config["versions"]:
        raise ValueError("versions.yaml 中 versions 为空，至少需要配置一个版本。")

    return config


def get_versions(config: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    """
    获取所有版本配置。
    """
    if config is None:
        config = load_versions_config()

    versions = config.get("versions", {})
    if not isinstance(versions, dict) or not versions:
        raise ValueError("versions.yaml 中 versions 配置无效。")

    return versions


def get_version_order(config: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    获取版本展示顺序。

    优先使用 version_order；
    如果没有 version_order，则使用 versions 的 key 顺序。
    """
    if config is None:
        config = load_versions_config()

    versions = get_versions(config)
    version_order = config.get("version_order")

    if not version_order:
        return list(versions.keys())

    # 过滤掉不存在的版本，避免 YAML 写错导致页面崩溃
    valid_order = [key for key in version_order if key in versions]

    # 如果 version_order 全部无效，则回退到 versions 原始顺序
    if not valid_order:
        return list(versions.keys())

    # 如果 versions 里有新版本但没写进 version_order，追加到最后
    for key in versions.keys():
        if key not in valid_order:
            valid_order.append(key)

    return valid_order


def get_default_version_key(config: Optional[Dict[str, Any]] = None) -> str:
    """
    获取默认版本 key。

    优先读取 default_version；
    如果 default_version 不存在或无效，则使用 version_order 的第一个版本。
    """
    if config is None:
        config = load_versions_config()

    versions = get_versions(config)
    version_order = get_version_order(config)

    default_version = config.get("default_version")

    if default_version in versions:
        return default_version

    return version_order[0]


def get_version_config(
    version_key: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    获取某个版本的配置。

    参数：
        version_key: 版本 key。为 None 时自动使用 default_version。
        config: 已读取的配置字典，可不传。

    返回：
        dict: 当前版本的配置。
    """
    if config is None:
        config = load_versions_config()

    versions = get_versions(config)

    if version_key is None:
        version_key = get_default_version_key(config)

    if version_key not in versions:
        default_key = get_default_version_key(config)
        version_key = default_key

    return versions[version_key]


def get_current_version_key(
    version_key: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    返回有效版本 key。

    如果传入的 version_key 无效，则返回默认版本 key。
    """
    if config is None:
        config = load_versions_config()

    versions = get_versions(config)

    if version_key in versions:
        return str(version_key)

    return get_default_version_key(config)


def get_version_files(
    version_key: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    获取当前版本对应的 files 配置。
    """
    version_cfg = get_version_config(version_key=version_key, config=config)
    files = version_cfg.get("files", {})

    if not isinstance(files, dict):
        return {}

    return files


# ============================================================
# Streamlit 页面通用版本选择控件
# ============================================================

def select_version_sidebar(
    label: str = "选择分型版本",
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    show_description: bool = True,
    show_method: bool = True,
    key: str = "selected_version_key",
) -> Tuple[str, Dict[str, Any], Dict[str, str]]:
    """
    在 Streamlit sidebar 中创建 M1-M5 版本选择器。

    返回：
        selected_version_key: 当前选择的版本 key
        current_version: 当前版本完整配置
        current_files: 当前版本 files 字典

    用法：
        selected_version, current_version, current_files = select_version_sidebar()
    """
    try:
        config = load_versions_config(config_path)
        versions = get_versions(config)
        version_order = get_version_order(config)
        default_version = get_default_version_key(config)
    except Exception as e:
        st.sidebar.error(f"版本配置读取失败：{e}")
        st.stop()

    default_index = 0
    if default_version in version_order:
        default_index = version_order.index(default_version)

    selected_version_key = st.sidebar.selectbox(
        label,
        version_order,
        index=default_index,
        key=key,
        format_func=lambda version_key: versions[version_key].get(
            "display_name",
            version_key,
        ),
    )

    current_version = versions[selected_version_key]
    current_files = current_version.get("files", {})
    if not isinstance(current_files, dict):
        current_files = {}

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**当前版本：** {current_version.get('short_name', selected_version_key)}"
    )

    role = current_version.get("role", "")
    if role:
        role_display = {
            "main": "主实验版本",
            "ablation": "消融对照版本",
        }.get(role, role)
        st.sidebar.markdown(f"**版本角色：** {role_display}")

    if show_method:
        st.sidebar.markdown(f"**当前方法：** {current_version.get('method', '-')}")

    if show_description:
        desc = current_version.get("description", "")
        if desc:
            st.sidebar.caption(desc)

    return selected_version_key, current_version, current_files


# ============================================================
# 文件路径工具
# ============================================================

def resolve_path(path_value: Optional[str | Path]) -> Optional[Path]:
    """
    将 YAML 中的路径转为 Path。

    说明：
    - 如果是绝对路径，例如 H:/xxx，则直接返回。
    - 如果是相对路径，例如 outputs/M1/xxx.csv，则基于项目根目录 ROOT_DIR 解析。
    - 如果为空，则返回 None。
    """
    if not path_value:
        return None

    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def get_file_path(
    file_key: str,
    version_key: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """
    获取当前版本中某个文件 key 对应的完整 Path。

    示例：
        get_file_path("consensus_labels", "M1_新公平_Attn_topk4")
    """
    files = get_version_files(version_key=version_key, config=config)
    raw_path = files.get(file_key)
    return resolve_path(raw_path)


def file_exists(
    file_key: str,
    version_key: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    判断某个版本配置中的文件是否存在。
    """
    path = get_file_path(file_key, version_key=version_key, config=config)
    return bool(path and path.exists())


def safe_read_csv(
    path_value: Optional[str | Path],
    encoding: str = "utf-8-sig",
) -> Optional[pd.DataFrame]:
    """
    安全读取 CSV。

    参数：
        path_value: CSV 路径，可以是绝对路径或相对路径。
        encoding: 默认 utf-8-sig，兼容中文 CSV。

    返回：
        DataFrame 或 None。
    """
    path = resolve_path(path_value)

    if path is None or not path.exists():
        return None

    try:
        return pd.read_csv(path, encoding=encoding)
    except UnicodeDecodeError:
        # 有些 CSV 可能是 gbk
        try:
            return pd.read_csv(path, encoding="gbk")
        except Exception:
            return None
    except Exception:
        return None


def safe_read_version_csv(
    file_key: str,
    version_key: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    encoding: str = "utf-8-sig",
) -> Optional[pd.DataFrame]:
    """
    读取某个版本配置下的 CSV 文件。

    示例：
        df = safe_read_version_csv("consensus_labels", selected_version)
    """
    path = get_file_path(file_key, version_key=version_key, config=config)

    if path is None:
        return None

    return safe_read_csv(path, encoding=encoding)


def get_umap_path(
    version_key: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """
    获取当前版本 UMAP 图路径。
    """
    return get_file_path("umap_figure", version_key=version_key, config=config)


# ============================================================
# 展示辅助函数
# ============================================================

def get_version_display_name(
    version_key: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    获取版本展示名。
    """
    valid_key = get_current_version_key(version_key, config)
    version_cfg = get_version_config(valid_key, config)
    return version_cfg.get("display_name", valid_key)


def get_version_short_name(
    version_key: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    获取版本短名，如 M1 / M2。
    """
    valid_key = get_current_version_key(version_key, config)
    version_cfg = get_version_config(valid_key, config)
    return version_cfg.get("short_name", valid_key)


def get_version_method(
    version_key: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    获取版本方法描述。
    """
    version_cfg = get_version_config(version_key, config)
    return version_cfg.get("method", "-")


def get_boundary_threshold(
    version_key: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> float:
    """
    获取当前版本边界患者阈值，默认 0.8。
    """
    version_cfg = get_version_config(version_key, config)
    try:
        return float(version_cfg.get("boundary_threshold", 0.8))
    except Exception:
        return 0.8


def get_n_clusters(
    version_key: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> int:
    """
    获取当前版本聚类数量，默认 3。
    """
    version_cfg = get_version_config(version_key, config)
    try:
        return int(version_cfg.get("n_clusters", 3))
    except Exception:
        return 3


def build_version_summary_row(
    version_key: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    构建一个版本摘要行，可用于版本对比表。
    """
    if config is None:
        config = load_versions_config()

    version_cfg = get_version_config(version_key, config)

    role = version_cfg.get("role", "")
    role_display = {
        "main": "主实验版本",
        "ablation": "消融对照版本",
    }.get(role, role)

    top_k = version_cfg.get("top_k", "-")
    if top_k == 0:
        top_k_display = "全部帧"
    else:
        top_k_display = top_k

    return {
        "版本": version_cfg.get("short_name", version_key),
        "显示名称": version_cfg.get("display_name", version_key),
        "角色": role_display,
        "抽帧策略": version_cfg.get("sampling_name", "-"),
        "Pooling": version_cfg.get("pooling", "-"),
        "top-k": top_k_display,
        "方法": version_cfg.get("method", "-"),
        "聚类数": version_cfg.get("n_clusters", 3),
        "边界阈值": version_cfg.get("boundary_threshold", 0.8),
    }


def build_versions_summary_table(
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    构建 M1-M5 版本配置摘要表。

    可在首页或方法对比页展示。
    """
    if config is None:
        config = load_versions_config()

    version_order = get_version_order(config)
    rows = [build_version_summary_row(v, config) for v in version_order]
    return pd.DataFrame(rows)


# ============================================================
# 页面调试 / 文件检查
# ============================================================

def check_version_files(
    version_key: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    检查某个版本 files 中所有文件是否存在。

    返回 DataFrame：
        file_key | path | exists
    """
    if config is None:
        config = load_versions_config()

    valid_key = get_current_version_key(version_key, config)
    files = get_version_files(valid_key, config)

    rows = []
    for file_key, raw_path in files.items():
        path = resolve_path(raw_path)
        rows.append(
            {
                "file_key": file_key,
                "path": str(path) if path else "",
                "exists": bool(path and path.exists()),
            }
        )

    return pd.DataFrame(rows)


def show_version_file_status(
    version_key: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
):
    """
    在 Streamlit 页面中展示当前版本文件检查结果。
    通常放在 expander 调试区域。
    """
    df = check_version_files(version_key=version_key, config=config)

    if df.empty:
        st.info("当前版本未配置 files。")
        return

    st.dataframe(df, use_container_width=True, hide_index=True)

    missing = df[df["exists"] == False]  # noqa: E712
    if not missing.empty:
        st.warning("存在未找到的配置文件，请检查 versions.yaml 中的路径。")
    else:
        st.success("当前版本配置文件均可找到。")


# ============================================================
# 缓存清理
# ============================================================

def clear_version_config_cache():
    """
    清除版本配置缓存。
    当你修改 versions.yaml 后，可以调用这个函数刷新。
    """
    load_versions_config.clear()