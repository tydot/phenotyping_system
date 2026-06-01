"""
患者视图（Patient View）
ARM 功能表型系统

说明：
1. 支持 M1-M5 侧边栏版本切换。
2. 分型结果来自当前版本 consensus_labels。
3. 临床指标来自当前版本 clinical_with_clusters / merged_clinical_all。
4. raw_row = 临床合并行 + 当前版本分型行。
5. LLM 解释层只解释结构化结果，不参与分型。
6. VLM / FR-GCD-Lite 只作为图像侧区域辅助解释，不参与分型，不修改 cluster。
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import streamlit.components.v1 as components
import tempfile
import os
import subprocess

from dotenv import load_dotenv, dotenv_values

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# 必须在导入 llm_client 之前读取 .env
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(ENV_PATH, override=True)

from backend.vlm.region_guided_decoder import generate_region_findings
from backend.vlm.consistency_gate import check_visual_clinical_consistency

env_values = dotenv_values(ENV_PATH)
for key in [
    "LLM_ENABLE_REAL_API",
    "LLM_PROVIDER",
    "XIAOMI_API_KEY",
    "XIAOMI_BASE_URL",
    "XIAOMI_MODEL",
    "MIMO_API_KEY",
    "MIMO_BASE_URL",
    "MIMO_MODEL",
]:
    value = env_values.get(key)
    if value is not None:
        os.environ[key] = str(value).strip()

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from backend.api.patient import get_patient_view
from backend.auth.auth_service import require_login, can_view_patient
from backend.version_manager import (
    select_version_sidebar,
    safe_read_csv,
    resolve_path,
)
from backend.report.feature_state_extractor import (
    extract_metric_judgements,
    extract_feature_states,
    debug_metric_mapping,
)
from backend.report.llm_report import build_llm_context
from backend.report.llm_client import generate_llm_report, get_llm_runtime_status
from pathlib import Path
import pandas as pd
import streamlit as st

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
    patient 页面统一读取当前版本患者联合表。
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


def normalize_patient_df(df: pd.DataFrame) -> pd.DataFrame:
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

    if "is_boundary" not in df.columns:
        if "confidence" in df.columns and df["confidence"].notna().any():
            df["is_boundary"] = df["confidence"].apply(
                lambda x: bool(pd.notna(x) and float(x) < 0.8)
            )
        else:
            df["is_boundary"] = pd.NA

    return df


def find_patient_row(df: pd.DataFrame, patient_id: str) -> pd.DataFrame:
    if df is None or df.empty or "patient_id" not in df.columns:
        return pd.DataFrame()

    target = normalize_patient_id(patient_id)

    matched = df[df["patient_id"].apply(normalize_patient_id) == target]

    return matched


# -----------------------------
# Knowledge Graph / Pyvis imports
# 允许部署环境缺少 neo4j 或 pyvis 依赖时，患者页主体仍能运行
# -----------------------------
GRAPH_FEATURE_AVAILABLE = True
GRAPH_IMPORT_ERROR = None
Network = None

try:
    from pyvis.network import Network
    from backend.graph.upsert_patient_graph import upsert_patient_knowledge_graph
    from backend.graph.patient_graph_pipeline import get_patient_graph_for_frontend
except Exception as e:
    GRAPH_FEATURE_AVAILABLE = False
    GRAPH_IMPORT_ERROR = str(e)


@st.cache_data(show_spinner=False)
def load_patient_view(patient_id: str):
    return get_patient_view(patient_id)


@st.cache_data(show_spinner=False)
def load_patient_graph(patient_id: str):
    if not GRAPH_FEATURE_AVAILABLE:
        return {"nodes": [], "edges": [], "paths": []}
    return get_patient_graph_for_frontend(patient_id)


@st.cache_data(show_spinner=False)
def load_version_csv(file_path_str: str):
    path = resolve_path(file_path_str)
    return safe_read_csv(path)


def clear_patient_cache():
    load_patient_view.clear()
    load_patient_graph.clear()
    load_version_csv.clear()


def safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def build_physiology_from_raw_row(raw_row: dict) -> dict:
    raw_row = safe_dict(raw_row)

    core_keys = [
        "肛门括约肌静息压(mmHg)",
        "最大缩榨压MSP（mmHg）",
        "排便时直肠压力(mmHg)",
        "RAIR诱发最小容积(ml)",
    ]

    desc_keys = [
        "肛门括约肌长度(cm)",
        "缩肛持续时间(s)",
        "初始感觉阈值(ml)",
        "初始便意阈值(ml)",
        "排便窘迫感阈值(ml)",
        "最大容量感觉阈值(ml)",
    ]

    core_metrics = {}
    descriptive_metrics = {}

    for key in core_keys:
        value = raw_row.get(key)
        if value is not None and str(value).strip() not in ["", "nan", "None", "-"]:
            core_metrics[key] = value

    for key in desc_keys:
        value = raw_row.get(key)
        if value is not None and str(value).strip() not in ["", "nan", "None", "-"]:
            descriptive_metrics[key] = value

    return {
        "core_metrics": core_metrics,
        "descriptive_metrics": descriptive_metrics,
    }


def add_unique_id(candidates: List[str], value: Any):
    value = normalize_patient_id(value)
    if value and value not in candidates:
        candidates.append(value)


def load_backend_patient_with_fallback(
    input_patient_id: str,
    patient_row: pd.Series,
):
    """
    后端 KG / RAG / LLM / VLM / RAIR / Rome 使用的患者对象。
    因为当前版本 CSV 里的 patient_id / pid_key 可能和后端患者库主键不同，
    所以这里尝试多个候选 ID。
    """
    candidates: List[str] = []

    add_unique_id(candidates, input_patient_id)

    if patient_row is not None:
        for key in [
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
        ]:
            if key in patient_row.index:
                add_unique_id(candidates, patient_row.get(key))

    debug_rows = []

    for candidate_id in candidates:
        backend_patient = load_patient_view(candidate_id) or {}

        if isinstance(backend_patient, dict):
            keys = list(backend_patient.keys())
        else:
            keys = []

        debug_rows.append(
            {
                "候选后端ID": candidate_id,
                "是否读到后端患者对象": bool(keys),
                "后端字段预览": ", ".join(keys[:12]) if keys else "-",
            }
        )

        if keys:
            return candidate_id, backend_patient, debug_rows

    fallback_id = candidates[0] if candidates else input_patient_id
    return fallback_id, {}, debug_rows


def fmt_number(x, digits: int = 2):
    if x is None or x == "":
        return "-"
    if isinstance(x, bool):
        return "是" if x else "否"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def dict_to_df(data: dict, col1: str = "指标", col2: str = "数值") -> pd.DataFrame:
    if not isinstance(data, dict) or not data:
        return pd.DataFrame(columns=[col1, col2])
    return pd.DataFrame(list(data.items()), columns=[col1, col2])


def show_metric_table(data: Dict[str, Any], digits: int = 2, empty_text: str = "暂无数据。"):
    df = dict_to_df(data)
    if df.empty:
        st.caption(empty_text)
        return
    df["数值"] = df["数值"].apply(lambda x: fmt_number(x, digits))
    st.dataframe(df, use_container_width=True, hide_index=True)


def normalize_patient_id(raw_value: Any) -> str:
    if raw_value is None:
        return ""

    try:
        if pd.isna(raw_value):
            return ""
    except Exception:
        pass

    s = str(raw_value).strip()

    if s.endswith(".0"):
        s = s[:-2]

    return s


def normalize_id_for_match(value: Any) -> str:
    """
    统一 patient_id 格式。
    解决 CSV 里 210259070 被 pandas 读成 210259070.0 后无法匹配的问题。
    """
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def normalize_bool_value(x):
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in ["true", "1", "yes", "y", "是", "边界", "boundary"]


def get_patient_version_row(
    patient_id: str,
    current_files: dict,
    boundary_threshold: float,
    current_version: Optional[dict] = None,
):
    """
    从当前选中版本中查找患者分型结果，并补充当前版本的 merged_clinical_all.csv 临床指标。

    支持 M1-M5 版本切换，并输出 clinical_debug 方便排查为什么临床文件没读到。
    """
    current_version = current_version or {}
    current_files = current_files or {}
    target_pid = normalize_id_for_match(patient_id)

    def find_patient_row(file_path: Optional[str]):
        if not file_path:
            return None, None, None

        df = load_version_csv(file_path)
        if df is None or df.empty:
            return None, None, None

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
                "pid_key",
            ],
        )

        if not patient_col:
            return None, df, None

        df = df.copy()
        df["__pid_match__"] = df[patient_col].apply(normalize_id_for_match)
        row_df = df[df["__pid_match__"] == target_pid]

        if row_df.empty:
            return None, df, patient_col

        row = row_df.iloc[0].drop(labels=["__pid_match__"], errors="ignore").to_dict()
        return row, df.drop(columns=["__pid_match__"], errors="ignore"), patient_col

    def get_cluster_info(row: dict, df: pd.DataFrame):
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

        return cluster_val, confidence_val, is_boundary_val

    def infer_merged_path_from_consensus(consensus_path: Optional[str]):
        if not consensus_path:
            return None

        p = str(consensus_path)
        if "consensus_labels_all.csv" in p:
            return p.replace("consensus_labels_all.csv", "merged_clinical_all.csv")
        if "consensus_labels.csv" in p:
            return p.replace("consensus_labels.csv", "merged_clinical_all.csv")
        return None

    short_name = current_version.get("short_name", "")
    consensus_file = current_files.get("consensus_labels")

    clinical_file = (
        current_files.get("clinical_with_clusters")
        or current_files.get("merged_clinical")
        or current_files.get("patient_clinical")
        or current_files.get("cohort_table")
    )

    merged_file = (
        current_files.get("merged_clinical")
        or current_files.get("patient_clinical")
        or current_files.get("cohort_table")
    )

    inferred_merged_file = infer_merged_path_from_consensus(consensus_file)

    clinical_candidate_files = [
        clinical_file,
        merged_file,
        inferred_merged_file,
    ]

    label_candidate_files = [
        consensus_file,
        clinical_file,
        merged_file,
        inferred_merged_file,
    ]

    # ------------------------------------------------------------
    # 1. 读取当前版本分型结果
    # ------------------------------------------------------------
    label_row = None
    label_df = None
    label_source_file = None

    for file_path in label_candidate_files:
        row, df, _ = find_patient_row(file_path)
        if row is None or df is None:
            continue

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

        if cluster_col or confidence_col:
            label_row = row
            label_df = df
            label_source_file = file_path
            break

    if label_row is None or label_df is None:
        return None

    cluster_val, confidence_val, is_boundary_val = get_cluster_info(label_row, label_df)

    # ------------------------------------------------------------
    # 2. 读取当前版本临床合并文件
    # ------------------------------------------------------------
    clinical_row = None
    clinical_source_file = None
    clinical_debug = []

    for file_path in clinical_candidate_files:
        if not file_path:
            clinical_debug.append(
                {
                    "候选文件": "-",
                    "状态": "空路径",
                    "字段数": "-",
                    "患者列": "-",
                    "是否找到患者": False,
                }
            )
            continue

        df = load_version_csv(file_path)

        if df is None:
            clinical_debug.append(
                {
                    "候选文件": file_path,
                    "状态": "读取失败或文件不存在",
                    "字段数": "-",
                    "患者列": "-",
                    "是否找到患者": False,
                }
            )
            continue

        if df.empty:
            clinical_debug.append(
                {
                    "候选文件": file_path,
                    "状态": "文件为空",
                    "字段数": 0,
                    "患者列": "-",
                    "是否找到患者": False,
                }
            )
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
                "pid_key",
            ],
        )

        if not patient_col:
            clinical_debug.append(
                {
                    "候选文件": file_path,
                    "状态": "未找到患者ID列",
                    "字段数": len(df.columns),
                    "患者列": "-",
                    "是否找到患者": False,
                }
            )
            continue

        df2 = df.copy()
        df2["__pid_match__"] = df2[patient_col].apply(normalize_id_for_match)
        row_df = df2[df2["__pid_match__"] == target_pid]
        found_patient = not row_df.empty

        clinical_debug.append(
            {
                "候选文件": file_path,
                "状态": "已读取",
                "字段数": len(df.columns),
                "患者列": patient_col,
                "是否找到患者": found_patient,
            }
        )

        if row_df.empty:
            continue

        row = row_df.iloc[0].drop(labels=["__pid_match__"], errors="ignore").to_dict()

        if file_path == label_source_file and len(row.keys()) <= 20:
            continue

        clinical_row = row
        clinical_source_file = file_path
        break

    # ------------------------------------------------------------
    # 3. 合并 raw_row
    # ------------------------------------------------------------
    if clinical_row:
        raw_row = dict(clinical_row)
        raw_row.update(label_row)
    else:
        raw_row = dict(label_row)

    raw_row["consensus_cluster"] = cluster_val
    raw_row["confidence"] = confidence_val
    raw_row["is_boundary"] = is_boundary_val
    raw_row["switch_rate"] = 1 - confidence_val if confidence_val is not None else None

    return {
        "raw_row": raw_row,
        "source_file": label_source_file,
        "clinical_source_file": clinical_source_file,
        "clinical_debug": clinical_debug,
        "cluster": cluster_val,
        "confidence": confidence_val,
        "is_boundary": is_boundary_val,
        "switch_rate": 1 - confidence_val if confidence_val is not None else None,
    }


def sync_patient_graph(patient_id: str, patient: Dict[str, Any]) -> str | None:
    """
    将当前患者数据写入 Neo4j，并清理图谱缓存。
    成功返回 None，失败返回错误信息。
    """
    if not GRAPH_FEATURE_AVAILABLE:
        return f"知识图谱模块不可用：{GRAPH_IMPORT_ERROR}"

    try:
        upsert_patient_knowledge_graph(patient, patient_id=patient_id)
        load_patient_graph.clear()
        return None
    except Exception as e:
        return str(e)


def short_text(text: Any, max_len: int = 14) -> str:
    text = "" if text is None else str(text)
    return text if len(text) <= max_len else text[:max_len] + "..."


def build_node_title(label: str, node_type: str, props: Dict[str, Any]) -> str:
    lines = [f"<b>{label}</b>", f"type: {node_type}"]
    for k, v in props.items():
        lines.append(f"{k}: {v}")
    return "<br>".join(lines)


def build_edge_title(relation: str, props: Dict[str, Any]) -> str:
    lines = [f"<b>{relation}</b>"]
    for k, v in props.items():
        lines.append(f"{k}: {v}")
    return "<br>".join(lines)


def select_graph_subset(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]):
    typed_nodes: Dict[str, List[Dict[str, Any]]] = {}
    for node in safe_list(nodes):
        node = safe_dict(node)
        ntype = str(node.get("type", "Unknown"))
        typed_nodes.setdefault(ntype, []).append(node)

    selected: List[Dict[str, Any]] = []
    keep_all_types = {"Patient", "Phenotype", "Mechanism", "Recommendation", "Cluster"}

    for ntype, items in typed_nodes.items():
        if ntype in keep_all_types:
            selected.extend(items)
        elif ntype == "Feature":
            selected.extend(items[:8])
        elif ntype == "Evidence":
            selected.extend(items[:4])
        else:
            selected.extend(items[:6])

    keep_ids = {safe_dict(n).get("id") for n in selected if safe_dict(n).get("id")}

    filtered_edges: List[Dict[str, Any]] = []
    for edge in safe_list(edges):
        edge = safe_dict(edge)
        source = edge.get("source")
        target = edge.get("target")
        if source in keep_ids and target in keep_ids:
            filtered_edges.append(edge)

    connected_ids = set()
    for edge in filtered_edges:
        connected_ids.add(edge.get("source"))
        connected_ids.add(edge.get("target"))

    filtered_nodes = [
        n for n in selected
        if safe_dict(n).get("id") in connected_ids
        or safe_dict(n).get("type") == "Patient"
    ]

    return filtered_nodes, filtered_edges


def render_knowledge_graph_pyvis(nodes, edges, height: int = 760):
    if not GRAPH_FEATURE_AVAILABLE or Network is None:
        st.info("当前环境未安装 pyvis / Neo4j 相关依赖，知识图谱可视化暂不可用。")
        if GRAPH_IMPORT_ERROR:
            st.caption(f"导入信息：{GRAPH_IMPORT_ERROR}")
        return

    display_nodes, display_edges = select_graph_subset(
        safe_list(nodes),
        safe_list(edges),
    )

    net = Network(
        height=f"{height}px",
        width="100%",
        bgcolor="#1f2430",
        font_color="white",
        directed=True,
    )

    type_style = {
        "Patient": {"color": "#c678dd", "size": 38},
        "Phenotype": {"color": "#56b6c2", "size": 30},
        "Feature": {"color": "#f4a261", "size": 22},
        "Mechanism": {"color": "#98c379", "size": 28},
        "Evidence": {"color": "#e06c75", "size": 20},
        "Recommendation": {"color": "#e5c07b", "size": 24},
        "Cluster": {"color": "#61afef", "size": 24},
    }

    for node in display_nodes:
        node = safe_dict(node)
        node_id = node.get("id")
        if not node_id:
            continue

        label = str(node.get("label", ""))
        node_type = str(node.get("type", "Unknown"))
        props = safe_dict(node.get("properties"))
        style = type_style.get(node_type, {"color": "#7f848e", "size": 18})

        if node_type == "Patient":
            show_label = short_text(label, 18)
        elif node_type in {"Phenotype", "Mechanism", "Recommendation", "Cluster"}:
            show_label = short_text(label, 16)
        else:
            show_label = short_text(label, 10)

        net.add_node(
            node_id,
            label=show_label,
            title=build_node_title(label, node_type, props),
            color=style["color"],
            size=style["size"],
        )

    for edge in display_edges:
        edge = safe_dict(edge)
        source = edge.get("source")
        target = edge.get("target")
        relation = str(edge.get("relation", ""))
        props = safe_dict(edge.get("properties"))

        if not source or not target:
            continue

        show_relation_label = relation not in {"HAS_FEATURE", "HAS_EVIDENCE"}

        net.add_edge(
            source,
            target,
            label=relation if show_relation_label else "",
            title=build_edge_title(relation, props),
            arrows="to",
        )

    net.set_options("""
    const options = {
      "nodes": {
        "shape": "dot",
        "borderWidth": 2,
        "borderWidthSelected": 3,
        "font": {
          "size": 18,
          "color": "white",
          "face": "arial"
        }
      },
      "edges": {
        "arrows": {
          "to": {
            "enabled": true,
            "scaleFactor": 0.7
          }
        },
        "color": {
          "color": "#cfd6e4",
          "highlight": "#ffffff",
          "inherit": false
        },
        "font": {
          "size": 12,
          "color": "white",
          "strokeWidth": 0,
          "align": "middle"
        },
        "smooth": {
          "enabled": true,
          "type": "dynamic"
        },
        "width": 1.5
      },
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -3800,
          "centralGravity": 0.28,
          "springLength": 155,
          "springConstant": 0.035,
          "damping": 0.92,
          "avoidOverlap": 0.2
        },
        "minVelocity": 0.75
      },
      "interaction": {
        "hover": true,
        "navigationButtons": true,
        "keyboard": true,
        "tooltipDelay": 120
      }
    }
    """)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
        net.save_graph(f.name)
        html_path = f.name

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    components.html(html, height=height + 20, scrolling=False)

    try:
        os.remove(html_path)
    except Exception:
        pass


def normalize_gender_value(value: Any) -> str | None:
    if value is None:
        return None

    s = str(value).strip()
    if not s:
        return None

    mapping = {
        "男": "male",
        "male": "male",
        "Male": "male",
        "M": "male",
        "m": "male",
        "1": "male",
        "女": "female",
        "female": "female",
        "Female": "female",
        "F": "female",
        "f": "female",
        "0": "female",
        "2": "female",
    }

    return mapping.get(s, s)


def resolve_gender_display(patient: Dict[str, Any], gender_meta: Dict[str, Any]) -> Dict[str, Any]:
    patient = safe_dict(patient)
    gender_meta = safe_dict(gender_meta)

    patient_gender = normalize_gender_value(patient.get("gender"))
    patient_sex = normalize_gender_value(patient.get("sex"))
    actual_gender = patient_gender or patient_sex

    meta_gender = normalize_gender_value(gender_meta.get("resolved_gender"))
    meta_source = gender_meta.get("source", "-")
    meta_defaulted = bool(gender_meta.get("is_defaulted", False))

    if actual_gender:
        source = meta_source
        if source in (None, "", "-", "default"):
            source = "clinical"
        return {
            "resolved_gender": actual_gender,
            "source": source,
            "is_defaulted": False,
        }

    if meta_gender:
        return {
            "resolved_gender": meta_gender,
            "source": meta_source or "-",
            "is_defaulted": meta_defaulted,
        }

    return {
        "resolved_gender": "-",
        "source": "-",
        "is_defaulted": True,
    }


# ============================================================
# VLM / FR-GCD-Lite 辅助函数
# ============================================================

# ============================================================
# VLM / FR-GCD-Lite 辅助函数
# ============================================================

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
NON_IMAGE_EXTENSIONS = {".npy", ".npz", ".csv", ".pkl", ".pt", ".pth", ".json", ".txt"}
@st.cache_resource(show_spinner=False)
def get_cos_client():
    try:
        from qcloud_cos import CosConfig, CosS3Client
    except Exception:
        return None

    secret_id = st.secrets.get("COS_SECRET_ID", os.environ.get("COS_SECRET_ID"))
    secret_key = st.secrets.get("COS_SECRET_KEY", os.environ.get("COS_SECRET_KEY"))
    region = st.secrets.get("COS_REGION", os.environ.get("COS_REGION"))

    if not secret_id or not secret_key or not region:
        return None

    config = CosConfig(
        Region=region,
        SecretId=secret_id,
        SecretKey=secret_key,
        Scheme="https",
    )
    return CosS3Client(config)


@st.cache_data(show_spinner=False)
def find_and_download_image_from_cos(patient_id: str, filename_or_path: str) -> Optional[str]:
    client = get_cos_client()
    if client is None:
        return None

    bucket = st.secrets.get("COS_BUCKET", os.environ.get("COS_BUCKET"))
    prefix = st.secrets.get("COS_PREFIX", os.environ.get("COS_PREFIX", "images")).strip("/")
    if not bucket or not filename_or_path:
        return None

    patient_id = normalize_patient_id(patient_id)
    if not patient_id:
        return None

    stem = Path(filename_or_path).stem
    cache_dir = ROOT_DIR / ".cache" / "cos_images" / patient_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 搜索 COS 中 images/{patient_id}/ 下匹配的图片
    search_prefix = f"{prefix}/{patient_id}/"
    marker = ""
    while True:
        try:
            response = client.list_objects(
                Bucket=bucket,
                Prefix=search_prefix,
                Marker=marker,
                MaxKeys=1000,
            )
        except Exception:
            return None

        contents = response.get("Contents", []) or []
        for obj in contents:
            key = obj.get("Key", "")
            key_stem = Path(key).stem
            suffix = Path(key).suffix.lower()

            if suffix not in IMAGE_EXTENSIONS:
                continue
            if stem not in key_stem and key_stem not in stem:
                continue

            local_path = cache_dir / Path(key).name
            if not local_path.exists():
                try:
                    client.download_file(Bucket=bucket, Key=key, DestFilePath=str(local_path))
                except Exception:
                    continue

            if local_path.exists():
                return str(local_path)

        if response.get("IsTruncated"):
            marker = response.get("NextMarker", "")
        else:
            break

    return None


@st.cache_data(show_spinner=False)
def list_and_download_protocol_images_from_cos(patient_id: str) -> List[Dict[str, Any]]:
    """
    当 backend_representation.protocol_topk_details 为空时，
    直接从 COS 的 images/{patient_id}/ 下找每个协议阶段的代表图。
    """

    client = get_cos_client()
    if client is None:
        return []

    bucket = st.secrets.get("COS_BUCKET", os.environ.get("COS_BUCKET"))
    prefix = st.secrets.get("COS_PREFIX", os.environ.get("COS_PREFIX", "images")).strip("/")

    if not bucket:
        return []

    patient_id = normalize_patient_id(patient_id)
    if not patient_id:
        return []

    search_prefix = f"{prefix}/{patient_id}/"
    cache_dir = ROOT_DIR / ".cache" / "cos_images" / patient_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, Any]] = {}
    marker = ""

    while True:
        try:
            response = client.list_objects(
                Bucket=bucket,
                Prefix=search_prefix,
                Marker=marker,
                MaxKeys=1000,
            )
        except Exception:
            return []

        contents = response.get("Contents", []) or []

        for obj in contents:
            key = obj.get("Key", "")
            suffix = Path(key).suffix.lower()

            if suffix not in IMAGE_EXTENSIONS:
                continue

            protocol = infer_protocol_from_text(key)
            if not protocol:
                continue

            if protocol in results:
                continue

            local_path = cache_dir / Path(key).name

            if not local_path.exists():
                try:
                    client.download_file(
                        Bucket=bucket,
                        Key=key,
                        DestFilePath=str(local_path),
                    )
                except Exception:
                    continue

            if local_path.exists():
                results[protocol] = {
                    "protocol": protocol,
                    "image_path": str(local_path),
                    "source_item": {
                        "cos_key": key,
                        "source": "cos_protocol_fallback",
                    },
                    "filename": Path(key).name,
                    "rank": None,
                    "score": None,
                    "weight": None,
                }

        is_truncated = str(response.get("IsTruncated", "")).lower() == "true"
        if not is_truncated:
            break

        marker = response.get("NextMarker") or ""
        if not marker:
            break

    ordered_results: List[Dict[str, Any]] = []

    for protocol in PROTOCOL_DISPLAY_ORDER:
        if protocol in results:
            ordered_results.append(results[protocol])

    for protocol, item in results.items():
        if protocol not in PROTOCOL_DISPLAY_ORDER:
            ordered_results.append(item)

    return ordered_results
    bucket = st.secrets.get("COS_BUCKET", os.environ.get("COS_BUCKET"))
    prefix = st.secrets.get("COS_PREFIX", os.environ.get("COS_PREFIX", "images")).strip("/")

    if not bucket:
        return None

    patient_id = normalize_patient_id(patient_id)
    stem = Path(str(filename_or_path)).stem

    if not patient_id or not stem:
        return None

    cache_dir = ROOT_DIR / ".cache" / "cos_images" / patient_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    search_prefix = f"{prefix}/{patient_id}/"
    marker = ""

    while True:
        try:
            response = client.list_objects(
                Bucket=bucket,
                Prefix=search_prefix,
                Marker=marker,
                MaxKeys=1000,
            )
        except Exception:
            return None

        contents = response.get("Contents", []) or []

        for obj in contents:
            key = obj.get("Key", "")
            suffix = Path(key).suffix.lower()

            if suffix not in IMAGE_EXTENSIONS:
                continue

            if Path(key).stem != stem:
                continue

            local_path = cache_dir / Path(key).name

            if local_path.exists():
                return str(local_path)

            try:
                client.download_file(
                    Bucket=bucket,
                    Key=key,
                    DestFilePath=str(local_path),
                )
                if local_path.exists():
                    return str(local_path)
            except Exception:
                continue

        is_truncated = str(response.get("IsTruncated", "")).lower() == "true"
        if not is_truncated:
            break

        marker = response.get("NextMarker") or ""
        if not marker:
            break

    return None

def _clean_path_value(value: Any) -> Optional[str]:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null", "-"}:
        return None

    return s


def _resolve_candidate_path(path_value: Any) -> Optional[str]:
    """
    只接受真实图片路径。
    明确排除 .npy / .npz / .csv / .pkl / .pt 等非图像文件。
    """
    s = _clean_path_value(path_value)
    if not s:
        return None

    suffix = Path(s).suffix.lower()

    if suffix in NON_IMAGE_EXTENSIONS:
        return None

    if suffix and suffix not in IMAGE_EXTENSIONS:
        return None

    p = Path(s)
    if p.exists() and p.suffix.lower() in IMAGE_EXTENSIONS:
        return str(p)

    p2 = ROOT_DIR / s
    if p2.exists() and p2.suffix.lower() in IMAGE_EXTENSIONS:
        return str(p2)

    # Windows 绝对路径只有在当前环境真实存在时才返回。
    # Streamlit Cloud 上 H:\... 不存在，不能直接返回，否则会阻断 COS fallback。
    if suffix in IMAGE_EXTENSIONS and (":" in s or s.startswith("\\\\")):
        return None

@st.cache_data(show_spinner=False)
def find_image_by_stem(image_root_dir: str, filename_or_path: str) -> Optional[str]:
    image_root_dir = _clean_path_value(image_root_dir)
    filename_or_path = _clean_path_value(filename_or_path)

    if not filename_or_path:
        return None

    stem = Path(filename_or_path).stem

    # 1. 先查本地目录。Cloud 上通常不存在，没关系。
    if image_root_dir:
        root = Path(image_root_dir)
        if root.exists():
            for ext in IMAGE_EXTENSIONS:
                candidate = root / f"{stem}{ext}"
                if candidate.exists():
                    return str(candidate)

            for ext in IMAGE_EXTENSIONS:
                try:
                    matches = list(root.rglob(f"{stem}{ext}"))
                except (FileNotFoundError, PermissionError):
                    matches = []
                if matches:
                    return str(matches[0])

    # 2. 本地找不到，再去腾讯云 COS：
    # COS 结构应为 images/{patient_id}/{protocol}/images/*.png
    patient_id_for_cos = st.session_state.get("active_patient_id", "")
    return find_and_download_image_from_cos(
        patient_id=patient_id_for_cos,
        filename_or_path=filename_or_path,
    )


@st.cache_data(show_spinner=False)
def find_images_by_patient_protocol(
    images_root: str, patient_id: str
) -> Dict[str, str]:
    """
    按患者 ID + 协议名搜索本地 images 目录。
    目录结构: images/{patient_id}/{protocol}/images/*.png
    返回 {protocol: image_path} 的映射。
    """
    images_root = _clean_path_value(images_root)
    if not images_root or not patient_id:
        return {}

    root = Path(images_root) / patient_id
    if not root.exists():
        return {}

    results = {}
    protocol_map = {
        "contraction": "Contraction",
        "cough": "Cough",
        "defecation": "Defecation",
        "restpressure": "RestPressure",
        "rest": "RestPressure",
        "rair": "rair",
    }

    for protocol_dir in root.iterdir():
        if not protocol_dir.is_dir():
            continue

        dir_name_lower = protocol_dir.name.lower()
        canonical = protocol_map.get(dir_name_lower, protocol_map.get(dir_name_lower.replace("_", ""), ""))
        if not canonical:
            canonical = protocol_dir.name

        if canonical in results:
            continue

        # 搜索 images/ 子目录
        img_subdir = protocol_dir / "images"
        search_dir = img_subdir if img_subdir.exists() else protocol_dir

        for ext in IMAGE_EXTENSIONS:
            matches = sorted(search_dir.rglob(f"*{ext}"))
            if matches:
                results[canonical] = str(matches[0])
                break

    return results


def resolve_patient_image_path(
    patient: Dict[str, Any],
    raw_row: Dict[str, Any],
    representation: Dict[str, Any],
    protocol_topk_details: List[Any],
    image_root_dir: Optional[str] = None,
) -> Optional[str]:
    """
    从多个来源寻找患者图像路径。

    优先级：
    1. 如果已有明确 PNG/JPG 路径，直接使用；
    2. 如果 top-k 里只有 .npy filename，则到 image_root_dir 中查找同名 PNG/JPG；
    3. 找不到则返回 None，不影响主模型展示。
    """

    image_path_keys = [
        "image_path",
        "vlm_input_path",
        "pressure_image_path",
        "arm_image_path",
        "heatmap_path",
        "frame_path",
        "topk_image_path",
        "png_path",
        "jpg_path",
        "jpeg_path",
        "img_path",
        "crop_path",
    ]

    filename_keys = [
        "filename",
        "file_name",
        "npy_path",
        "feature_path",
        "path",
    ]

    sources = [
        safe_dict(raw_row),
        safe_dict(patient),
        safe_dict(representation),
    ]

    # 1. 先找明确图片路径
    for source in sources:
        for key in image_path_keys:
            candidate = _resolve_candidate_path(source.get(key))
            if candidate:
                return candidate

    for item in safe_list(protocol_topk_details):
        item = safe_dict(item)
        for key in image_path_keys:
            candidate = _resolve_candidate_path(item.get(key))
            if candidate:
                return candidate

    # 2. 再用 top-k 的 .npy 文件名去预处理图像目录找同名 png/jpg
    if image_root_dir:
        for item in safe_list(protocol_topk_details):
            item = safe_dict(item)

            for key in filename_keys:
                raw_name = item.get(key)
                if not raw_name:
                    continue

                matched_image = find_image_by_stem(
                    image_root_dir=image_root_dir,
                    filename_or_path=str(raw_name),
                )

                if matched_image:
                    return matched_image

    return None
PROTOCOL_DISPLAY_ORDER = [
    "RestPressure",
    "Contraction",
    "Defecation",
    "rair",
    "Cough",
]


def infer_protocol_from_text(value: Any) -> Optional[str]:
    """
    从协议名、文件名或路径文本中推断 ARM 协议阶段。
    """
    text = str(value or "").lower()

    if "restpressure" in text or "静息" in text:
        return "RestPressure"

    if (
        "contraction" in text
        or "squeeze" in text
        or "提肛" in text
        or "缩榨" in text
        or "压肛" in text
    ):
        return "Contraction"

    if "defecation" in text or "排便" in text:
        return "Defecation"

    if "rair" in text:
        return "rair"

    if "cough" in text or "咳嗽" in text:
        return "Cough"

    # 最后再做宽松 rest 匹配，避免误伤其它路径文本
    if "rest" in text:
        return "RestPressure"

    return None


def resolve_patient_image_paths_by_protocol(
    patient: Dict[str, Any],
    raw_row: Dict[str, Any],
    representation: Dict[str, Any],
    protocol_topk_details: List[Any],
    image_root_dir: Optional[str] = None,
    patient_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    为同一患者解析多个协议阶段的图像。

    当前策略：
    - 每个协议阶段最多取 1 张图；
    - 优先使用 top-k 关键帧中的 filename；
    - 用 filename 的 stem 去 preprocessed_features 中匹配同名 .png/.jpg；
    - 找不到则跳过，不影响主模型展示。
    """

    image_path_keys = [
        "image_path",
        "vlm_input_path",
        "pressure_image_path",
        "arm_image_path",
        "heatmap_path",
        "frame_path",
        "topk_image_path",
        "png_path",
        "jpg_path",
        "jpeg_path",
        "img_path",
        "crop_path",
    ]

    filename_keys = [
        "filename",
        "file_name",
        "npy_path",
        "feature_path",
        "path",
    ]

    results: Dict[str, Dict[str, Any]] = {}

    # 1. 优先从 protocol_topk_details 里找每个协议的一张代表图
    for item in safe_list(protocol_topk_details):
        item = safe_dict(item)

        protocol_text = " ".join(
            [
                str(item.get("protocol", "")),
                str(item.get("filename", "")),
                str(item.get("path", "")),
                str(item.get("image_path", "")),
                str(item.get("frame_path", "")),
            ]
        )

        protocol = infer_protocol_from_text(protocol_text)
        if not protocol:
            continue

        # 每个协议阶段只取第一张代表图
        if protocol in results:
            continue

        matched_image = None

        # 1.1 已经有明确图片路径
        for key in image_path_keys:
            matched_image = _resolve_candidate_path(item.get(key))
            if matched_image:
                break

        # 1.2 只有 .npy filename，则去预处理图像目录找同名 .png/.jpg
        if not matched_image and image_root_dir:
            for key in filename_keys:
                raw_name = item.get(key)
                if not raw_name:
                    continue

                matched_image = find_image_by_stem(
                    image_root_dir=image_root_dir,
                    filename_or_path=str(raw_name),
                )

                if matched_image:
                    break

        if matched_image:
            results[protocol] = {
                "protocol": protocol,
                "image_path": matched_image,
                "source_item": item,
                "filename": item.get("filename")
                or item.get("path")
                or Path(matched_image).name,
                "rank": item.get("rank"),
                "score": item.get("score"),
                "weight": item.get("weight"),
            }

    # 2. 兜底：如果 top-k 里没有可匹配图片，就沿用原来的单图逻辑
    if not results:
        fallback_image = resolve_patient_image_path(
            patient=patient,
            raw_row=raw_row,
            representation=representation,
            protocol_topk_details=protocol_topk_details,
            image_root_dir=image_root_dir,
        )

        if fallback_image:
            protocol = infer_protocol_from_text(fallback_image) or "Unknown"
            results[protocol] = {
                "protocol": protocol,
                "image_path": fallback_image,
                "source_item": {},
                "filename": Path(fallback_image).name,
                "rank": None,
                "score": None,
                "weight": None,
            }

    # 3. 兜底：按患者 ID + 协议名直接搜索本地 images 目录
    if not results and image_root_dir and patient_id:
        local_images = find_images_by_patient_protocol(image_root_dir, patient_id)
        for proto, img_path in local_images.items():
            if proto not in results:
                results[proto] = {
                    "protocol": proto,
                    "image_path": img_path,
                    "source_item": {"source": "local_directory_scan"},
                    "filename": Path(img_path).name,
                    "rank": None,
                    "score": None,
                    "weight": None,
                }

    ordered_results: List[Dict[str, Any]] = []

    for protocol in PROTOCOL_DISPLAY_ORDER:
        if protocol in results:
            ordered_results.append(results[protocol])

    for protocol, value in results.items():
        if protocol not in PROTOCOL_DISPLAY_ORDER:
            ordered_results.append(value)

    return ordered_results
def build_vlm_feature_state_dict(
    feature_states: Any,
    metric_judgements: Any,
) -> Dict[str, Any]:
    """
    把 feature_states / metric_judgements 统一转换为 consistency_gate 可用的 dict。

    原因：
    - consistency_gate.check_visual_clinical_consistency 当前按 dict.items() 读取；
    - extract_feature_states() 的实际返回可能是 list / dict / 其他结构；
    - metric_judgements 中包含 status、state_text，更适合做图像-临床一致性门控。
    """
    result: Dict[str, Any] = {}

    if isinstance(feature_states, dict):
        for k, v in feature_states.items():
            result[str(k)] = v

    elif isinstance(feature_states, list):
        for idx, item in enumerate(feature_states):
            if isinstance(item, dict):
                key = (
                    item.get("feature")
                    or item.get("metric")
                    or item.get("name")
                    or item.get("指标")
                    or item.get("key")
                    or f"feature_state_{idx}"
                )
                result[str(key)] = item
            else:
                result[f"feature_state_{idx}"] = item

    elif feature_states:
        result["feature_states"] = feature_states

    for idx, item in enumerate(safe_list(metric_judgements)):
        item = safe_dict(item)
        metric = (
            item.get("metric")
            or item.get("指标")
            or item.get("name")
            or f"metric_{idx}"
        )
        status = item.get("status") or item.get("状态") or ""
        state_text = item.get("state_text") or item.get("解释") or ""
        value = item.get("value") or item.get("患者数值") or ""

        result[f"metric_{metric}"] = {
            "metric": metric,
            "status": status,
            "state_text": state_text,
            "value": value,
            "raw": item,
        }

    return result


def split_image_region_findings_for_llm(
    image_region_findings: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    usable = []
    uncertain_or_conflict = []

    for item in safe_list(image_region_findings):
        item = safe_dict(item)
        if item.get("use_in_report") is True:
            usable.append(item)
        else:
            uncertain_or_conflict.append(item)

    return {
        "usable": usable,
        "uncertain_or_conflict": uncertain_or_conflict,
    }

def show_image_safely(image_path: Any, title: str, width: int = 420):
    """
    用 PIL 读取本地图片后再交给 Streamlit 显示。
    比直接 st.image(path) 更稳，尤其是 Windows 路径、中文路径、长路径场景。
    """
    path_str = _clean_path_value(image_path)

    if not path_str:
        st.caption(f"{title}：路径为空。")
        return

    path_obj = Path(path_str)

    if not path_obj.exists():
        st.warning(f"{title} 文件不存在：{path_obj}")
        return

    try:
        img = Image.open(path_obj).convert("RGB")
        st.image(
            img,
            caption=f"{title} | {path_obj.name}",
            width=width,
        )
        st.caption(str(path_obj))
        st.caption(f"图像尺寸：{img.size[0]} × {img.size[1]}")
    except Exception as e:
        st.warning(f"{title} 显示失败：{e}")
        st.caption(str(path_obj))
def render_image_region_findings_section(
    image_path_for_vlm: Optional[str],
    image_region_findings: List[Dict[str, Any]],
    image_region_error: Optional[str],
    is_patient_user: bool,
):
    st.subheader("🖼️ 图像侧区域引导解释（FR-GCD-Lite）")
    st.caption(
        "该模块仅作为图像侧辅助证据，不参与无监督分型，不修改 cluster，不输出临床诊断。"
    )

    tab_region, tab_region_debug = st.tabs(["区域解释结果", "调试信息"])

    with tab_region:
        if image_path_for_vlm:
            st.caption(f"当前使用图像来源：{image_path_for_vlm}")
        else:
            st.info("当前患者未找到可用 ARM 图像路径，暂不生成图像侧区域解释。")

        if image_region_error:
            st.warning(f"图像侧区域解释生成失败：{image_region_error}")
        elif not image_path_for_vlm:
            st.info(
                "当前患者没有解析到可裁剪的 PNG/JPG 图像路径。"
                "请确认侧边栏中的 ARM 预处理图像文件夹是否正确，"
                "并确认该目录下存在与 top-k filename 同名的 .png 文件。"
            )

        if not image_region_findings:
            st.caption("暂无图像侧区域解释结果。")
        else:
            for item in image_region_findings:
                item = safe_dict(item)

                st.markdown(
                    f"### {item.get('region_name', item.get('region_id', '未知区域'))}"
                )

                # ------------------------------------------------------------
                # 图像可视化：同时显示 source_image_path 和 crop_path
                # ------------------------------------------------------------
                source_image_path = item.get("source_image_path")
                crop_path = item.get("crop_path")

                img_col1, img_col2 = st.columns(2)

                with img_col1:
                    st.markdown("**原始预处理图像**")
                    show_image_safely(
                        image_path=source_image_path,
                        title="source_image_path",
                        width=420,
                    )

                with img_col2:
                    st.markdown("**FR-GCD-Lite 输入图像**")
                    show_image_safely(
                        image_path=crop_path,
                        title="crop_path",
                        width=420,
                    )
            

                # ------------------------------------------------------------
                # 图像侧解释结果
                # ------------------------------------------------------------
                st.markdown("**图像侧解释结果**")

                visual_support = item.get("visual_support", "-")
                confidence = item.get("confidence", "-")
                consistency_status = item.get("consistency_status", "-")
                use_in_report = item.get("use_in_report", False)

                info_col1, info_col2 = st.columns(2)

                with info_col1:
                    st.write("**图像形态描述：**", item.get("visual_morphology", "-"))
                    st.write("**图像侧判断：**", visual_support)
                    st.write("**发现：**", item.get("finding", "-"))
                    st.write("**图像证据：**", item.get("evidence", "-"))
                with info_col2:
                    st.write("**置信度：**", fmt_number(confidence, 2))
                    st.write("**一致性状态：**", consistency_status)
                    st.write("**是否进入 LLM 报告：**", "是" if use_in_report else "否")

                note = item.get("consistency_note")
                if note:
                    if use_in_report:
                        st.success(note)
                    else:
                        st.info(note)

                st.divider()

    with tab_region_debug:
        if is_patient_user:
            st.caption("患者角色不展示图像侧调试信息。")
        else:
            st.write("image_path_for_vlm：", image_path_for_vlm or "-")
            st.write("image_region_error：", image_region_error or "-")
            st.json(image_region_findings)


# ============================================================
# Streamlit 页面开始
# ============================================================

st.set_page_config(page_title="患者视图 | ARM 功能表型系统", layout="wide")

st.title("🧠 患者功能表型视图")
st.caption("基于人工智能的功能亚型分配与生理证据展示，仅用于科研与表型探索")
st.divider()

selected_version, current_version, current_files = select_version_sidebar(
    key="patient_selected_version"
)

st.info(
    f"当前表型版本：**{current_version.get('display_name', selected_version)}**  \n"
    f"方法配置：**{current_version.get('method', '-')}**"
)

clinical_path_raw, clinical_path, patient_df_raw = load_current_version_clinical(current_files)
patient_df = normalize_patient_df(patient_df_raw)

with st.expander("调试：查看 patient 数据路径"):
    st.write("clinical 原始路径：", clinical_path_raw)
    st.write("clinical 解析路径：", str(clinical_path) if clinical_path else None)
    st.write("文件是否存在：", clinical_path.exists() if clinical_path else False)
    st.write("是否读取成功：", patient_df is not None and not patient_df.empty)

    if patient_df_raw is not None and not patient_df_raw.empty:
        st.write("原始列名：", list(patient_df_raw.columns))
        st.write("前 5 行：")
        st.dataframe(patient_df_raw.head(), use_container_width=True)

    if patient_df is not None and not patient_df.empty:
        st.write("标准化后列名：", list(patient_df.columns))
        st.write("前 30 个患者 ID：", patient_df["patient_id"].head(30).tolist())

if patient_df is None or patient_df.empty:
    st.error("当前版本未读取到患者联合表。请检查 versions.yaml 中 patient_clinical / cohort_table / merged_clinical / clinical_with_clusters 路径。")
    st.stop()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("患者选择")

available_patient_ids = (
    patient_df["patient_id"]
    .dropna()
    .astype(str)
    .map(normalize_patient_id)
    .loc[lambda s: s != ""]
    .drop_duplicates()
    .sort_values()
    .tolist()
)

if "active_patient_id" not in st.session_state:
    st.session_state["active_patient_id"] = ""

manual_patient_id = st.sidebar.text_input(
    "手动输入患者编号",
    value=st.session_state.get("active_patient_id", ""),
    placeholder="例如：190022372 或 210259070",
    key=f"patient_manual_input_{selected_version}",
)

if st.sidebar.button("确定加载患者", key=f"confirm_patient_{selected_version}"):
    st.session_state["active_patient_id"] = normalize_patient_id(manual_patient_id)
    clear_patient_cache()
    st.cache_data.clear()
    st.rerun()

patient_id = normalize_patient_id(st.session_state.get("active_patient_id", ""))

if not patient_id:
    st.info("请在左侧输入患者编号，并点击「确定加载患者」。")
    st.stop()

if st.sidebar.button("刷新患者数据", key=f"refresh_patient_{selected_version}_{patient_id}"):
    clear_patient_cache()
    st.cache_data.clear()
    st.rerun()

default_image_root = os.environ.get(
    "PREPROCESSED_FEATURES_DIR",
    str(ROOT_DIR / "images"),
)

image_root_input = st.sidebar.text_input(
    "ARM 预处理图像文件夹",
    value=default_image_root,
    placeholder="例如：preprocessed_features 或 /mount/src/phenotyping_system/preprocessed_features",
)

patient_row_df = find_patient_row(patient_df, patient_id)

if patient_row_df.empty:
    st.error("未找到该患者，或当前版本未返回有效数据。")

    st.warning(
        "请确认左侧输入的患者编号存在于当前版本的 merged_clinical_all.csv 中。"
    )

    st.write("当前版本可用患者编号示例：")
    st.dataframe(
        pd.DataFrame({"patient_id": available_patient_ids[:100]}),
        use_container_width=True,
        hide_index=True,
    )

    st.stop()

patient_row = patient_row_df.iloc[0]

# -----------------------------
# Auth / Permission
# -----------------------------
user = require_login()
role = user.get("role")
is_patient_user = role == "patient"

if not can_view_patient(user, patient_id):
    st.error("您无权限查看该患者。")
    st.stop()

# -----------------------------
# Load additional patient data from backend
# -----------------------------
backend_patient_id, patient, backend_patient_debug = load_backend_patient_with_fallback(
    input_patient_id=patient_id,
    patient_row=patient_row,
)
# ============================================================
# 后端固定结果：KG / RAG / LLM / VLM / RAIR / Rome / representation
# 这些不随 M1-M5 版本切换变化
# ============================================================

backend_ai = safe_dict(patient.get("ai_result"))
backend_phys = safe_dict(patient.get("physiology"))
backend_representation = safe_dict(patient.get("representation"))
backend_rair = safe_dict(patient.get("rair"))
backend_stab = safe_dict(patient.get("stability"))
backend_rome = safe_dict(patient.get("rome_iv"))
backend_group_stats = safe_dict(patient.get("group_statistics"))
backend_llm_analysis = safe_dict(patient.get("llm_analysis"))
backend_rag = safe_dict(patient.get("rag"))
backend_rag_recommendations = safe_list(patient.get("rag_recommendations"))
backend_gender_meta = safe_dict(patient.get("gender_meta"))

# 页面上方 AI 分型允许被当前版本覆盖
ai = dict(backend_ai)
stab = dict(backend_stab)
group_stats = dict(backend_group_stats)

# 下游解释模块固定走后端
phys = backend_phys
representation = backend_representation
rair = backend_rair
rome = backend_rome
llm_analysis = backend_llm_analysis
rag = backend_rag
rag_recommendations = backend_rag_recommendations
gender_meta = backend_gender_meta

# ============================================================
# 当前 M1-M5 版本结果覆盖
# ============================================================

boundary_threshold = current_version.get("boundary_threshold", 0.8)
try:
    boundary_threshold = float(boundary_threshold)
except Exception:
    boundary_threshold = 0.8

version_patient_result = get_patient_version_row(
    patient_id=patient_id,
    current_files=current_files,
    boundary_threshold=boundary_threshold,
    current_version=current_version,
)

if version_patient_result:
    ai["cluster"] = version_patient_result.get("cluster")
    ai["confidence"] = version_patient_result.get("confidence")
    ai["is_boundary"] = version_patient_result.get("is_boundary")

    stab["confidence"] = version_patient_result.get("confidence")
    stab["switch_rate"] = version_patient_result.get("switch_rate")
    stab["is_boundary"] = version_patient_result.get("is_boundary")
    stab["label"] = "边界患者" if version_patient_result.get("is_boundary") else "稳定患者"

    group_stats["version"] = current_version.get("display_name", selected_version)

    with st.expander("调试：查看当前版本患者分型来源"):
        st.write("当前版本：", current_version.get("display_name", selected_version))
        st.write("来源文件：", version_patient_result.get("source_file"))
        st.write("临床合并文件：", version_patient_result.get("clinical_source_file", "-"))
        st.write("Cluster：", version_patient_result.get("cluster"))
        st.write("Confidence：", version_patient_result.get("confidence"))
        st.write("是否边界：", version_patient_result.get("is_boundary"))

        raw_row = version_patient_result.get("raw_row", {})
        st.write("raw_row 字段数量：", len(raw_row))
        st.write("raw_row 字段列表：", list(raw_row.keys()))

        clinical_debug = version_patient_result.get("clinical_debug", [])
        if clinical_debug:
            st.markdown("**临床文件读取诊断**")
            clinical_debug_df = pd.DataFrame(clinical_debug).astype(str)

            st.dataframe(
                clinical_debug_df,
                use_container_width=True,
                hide_index=True,
            )

        show_raw_row = st.checkbox(
            "显示 raw_row 原始内容",
            value=False,
            key=f"show_raw_row_{patient_id}_{selected_version}",
        )
        if show_raw_row:
            st.json(raw_row)

else:
    st.warning(
        "当前版本结果文件中未找到该患者，将使用后端默认患者结果展示。"
    )
# ============================================================
# 云端 fallback：
# 如果 get_patient_view() 没有读到后端 patient 对象，
# 则至少用当前版本 CSV 的 raw_row / ai / stability 构造一个页面可用 patient。
# ============================================================

if (not patient) and version_patient_result:
    fallback_raw_row = safe_dict(version_patient_result.get("raw_row"))

    patient = {
        "patient_id": patient_id,
        "raw_row": fallback_raw_row,
        "ai_result": ai,
        "stability": stab,
        "physiology": build_physiology_from_raw_row(fallback_raw_row),
        "representation": {},
        "rair": {},
        "rome_iv": {},
        "rag": {},
        "rag_recommendations": [],
        "llm_analysis": {},
        "gender_meta": {},
    }

    backend_patient_id = patient_id

    backend_ai = safe_dict(patient.get("ai_result"))
    backend_phys = safe_dict(patient.get("physiology"))
    backend_representation = safe_dict(patient.get("representation"))
    backend_rair = safe_dict(patient.get("rair"))
    backend_stab = safe_dict(patient.get("stability"))
    backend_rome = safe_dict(patient.get("rome_iv"))
    backend_group_stats = safe_dict(patient.get("group_statistics"))
    backend_llm_analysis = safe_dict(patient.get("llm_analysis"))
    backend_rag = safe_dict(patient.get("rag"))
    backend_rag_recommendations = safe_list(patient.get("rag_recommendations"))
    backend_gender_meta = safe_dict(patient.get("gender_meta"))

    phys = backend_phys
    representation = backend_representation
    rair = backend_rair
    rome = backend_rome
    llm_analysis = backend_llm_analysis
    rag = backend_rag
    rag_recommendations = backend_rag_recommendations
    gender_meta = backend_gender_meta

    st.info("云端未读到后端 patient 对象，当前已使用版本 CSV 构造 fallback patient。")
# -----------------------------
# AI phenotype assignment
# -----------------------------
st.subheader("AI 表型分配结果（Phenotype Assignment）")
col1, col2, col3 = st.columns(3)

cluster_val = ai.get("cluster")
confidence_val = ai.get("confidence")
is_boundary = bool(ai.get("is_boundary", False))

with col1:
    st.metric("AI 分型（Cluster）", f"Cluster {cluster_val}" if cluster_val is not None else "未知")
with col2:
    st.metric("稳定性置信度（Confidence）", fmt_number(confidence_val, 2))
with col3:
    st.metric("分型稳定性（Stability）", "⚠️ 边界患者" if is_boundary else "稳定分配")

if is_boundary:
    st.warning("该患者位于表型分界区域，其分型结果在不同随机初始化下可能存在变化。")
else:
    st.caption("该患者的表型分配在多随机种子下保持稳定。")

st.caption("稳定性置信度反映多随机种子无监督聚类结果的一致性，并不等同于临床诊断置信度。")

gender_patient_for_display = dict(patient) if isinstance(patient, dict) else {}

raw_row_for_gender = {}
if version_patient_result and isinstance(version_patient_result.get("raw_row"), dict):
    raw_row_for_gender = version_patient_result.get("raw_row", {})

if not gender_patient_for_display.get("gender") and not gender_patient_for_display.get("sex"):
    for gender_key in ["性别", "gender", "sex"]:
        gender_value = raw_row_for_gender.get(gender_key)
        if gender_value is not None and str(gender_value).strip() not in ["", "nan", "None", "-"]:
            gender_patient_for_display["gender"] = gender_value
            break

effective_gender_info = resolve_gender_display(gender_patient_for_display, gender_meta)
resolved_gender = effective_gender_info.get("resolved_gender", "-")
gender_source = effective_gender_info.get("source", "-")
gender_is_defaulted = bool(effective_gender_info.get("is_defaulted", False))

if resolved_gender != "-":
    if gender_is_defaulted:
        st.warning(f"当前性别参考值为 {resolved_gender}，来源：{gender_source}。数据库未提供 gender，当前使用默认值。")
    else:
        st.caption(f"性别参考值：{resolved_gender}（来源：{gender_source}）")
else:
    st.warning("当前未获取到患者性别信息，系统将无法可靠匹配性别参考值。")

st.divider()

# -----------------------------
# Physiology evidence
# -----------------------------
st.subheader("生理证据（ARM 功能指标）")
core_metrics = safe_dict(phys.get("core_metrics"))
desc_metrics = safe_dict(phys.get("descriptive_metrics"))

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**核心功能指标（Definition Axes）**")
    show_metric_table(core_metrics, digits=2, empty_text="暂无核心 ARM 功能指标。")

with col_right:
    st.markdown("**描述性指标（Characterization Axes）**")
    show_metric_table(desc_metrics, digits=2, empty_text="暂无描述性 ARM 指标。")

st.markdown("**医院报告参考范围判定**")

metric_judgements = []
feature_states = []

if version_patient_result and "raw_row" in version_patient_result:
    metric_judgements = extract_metric_judgements(version_patient_result["raw_row"])
    feature_states = extract_feature_states(version_patient_result["raw_row"])

    judge_df = pd.DataFrame(metric_judgements)

    if not judge_df.empty:
        judge_df = judge_df.rename(
            columns={
                "metric": "指标",
                "value": "患者数值",
                "sex": "性别",
                "status": "状态",
                "low": "参考下限",
                "high": "参考上限",
                "center": "参考中心",
                "reference_group": "参考范围来源",
                "state_text": "解释",
            }
        )

        show_cols = [
            "指标",
            "患者数值",
            "性别",
            "状态",
            "参考下限",
            "参考上限",
            "参考中心",
            "参考范围来源",
            "解释",
        ]

        show_cols = [c for c in show_cols if c in judge_df.columns]

        st.dataframe(
            judge_df[show_cols],
            use_container_width=True,
            hide_index=True,
        )

        if not is_patient_user:
            with st.expander("调试：医院指标列名匹配情况"):
                debug_rows = debug_metric_mapping(version_patient_result["raw_row"])
                debug_df = pd.DataFrame(debug_rows)
                st.dataframe(debug_df, use_container_width=True, hide_index=True)
    else:
        st.caption("当前患者暂无可判定的医院参考范围指标。")
else:
    st.caption("当前版本结果文件中暂无患者原始临床指标，无法进行医院参考范围判定。")

# ============================================================
# ============================================================
# VLM / FR-GCD-Lite：图像侧区域引导解释
# ============================================================

st.divider()
st.subheader("🖼️ 图像侧区域引导解释（FR-GCD-Lite）")
st.caption(
    "该模块仅作为图像侧辅助证据，不参与无监督分型，不修改 cluster，不输出临床诊断。"
)

# -----------------------------
# 用户控制
# -----------------------------
enable_vlm = st.checkbox(
    "启用图像侧解释（调用 VLM API）",
    value=False,
    key=f"enable_vlm_{patient_id}",
)

force_refresh_vlm = st.checkbox(
    "强制刷新 VLM 缓存",
    value=False,
    key=f"force_refresh_vlm_{patient_id}",
)

# -----------------------------
# VLM cache 包装
# -----------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def cached_generate_region_findings(patient_id: str, image_path: str):
    return generate_region_findings(
        patient_id=patient_id,
        image_path=image_path,
        output_dir=str(ROOT_DIR / "outputs" / "vlm_region_crops"),
    )

# -----------------------------
# 解析五协议图像
# -----------------------------
raw_row_for_vlm = safe_dict(patient.get("raw_row")) or {}
protocol_topk_details_for_vlm = safe_list(backend_representation.get("protocol_topk_details"))

image_paths_for_vlm = resolve_patient_image_paths_by_protocol(
    patient=patient,
    raw_row=raw_row_for_vlm,
    representation=backend_representation,
    protocol_topk_details=protocol_topk_details_for_vlm,
    image_root_dir=image_root_input,
    patient_id=patient_id,
)

# COS fallback
if not image_paths_for_vlm:
    image_paths_for_vlm = list_and_download_protocol_images_from_cos(patient_id=patient_id)

# -----------------------------
# 展示调试信息
# -----------------------------
with st.expander("调试：VLM 图像路径解析"):
    st.write("解析到协议图数量：", len(image_paths_for_vlm))
    debug_rows = []
    for item in image_paths_for_vlm:
        debug_rows.append({
            "protocol": item.get("protocol"),
            "image_path": item.get("image_path"),
            "filename": item.get("filename"),
            "rank": item.get("rank"),
            "score": item.get("score"),
        })
    if debug_rows:
        st.dataframe(pd.DataFrame(debug_rows), use_container_width=True, hide_index=True)

# -----------------------------
# 展示五协议原图
# -----------------------------
if image_paths_for_vlm:
    st.markdown("### 五协议原始图像")
    cols = st.columns(2)
    for idx, item in enumerate(image_paths_for_vlm):
        protocol = item.get("protocol", "Unknown")
        image_path = item.get("image_path")
        with cols[idx % 2]:
            st.markdown(f"#### {protocol}")
            show_image_safely(image_path=image_path, title=protocol, width=420)
else:
    st.warning("当前患者未解析到可用协议图像。")

# -----------------------------
# 按钮触发 VLM
# -----------------------------
image_region_findings = []
image_region_error = None

if enable_vlm:
    if st.button("生成图像区域解释"):
        image_region_findings = []
        image_region_error = None
        try:
            all_region_findings = []
            progress_bar = st.progress(0)
            total_images = len(image_paths_for_vlm)

            for idx, path_item in enumerate(image_paths_for_vlm):
                protocol = path_item.get("protocol")
                one_image_path = path_item.get("image_path")
                st.info(f"正在分析协议：{protocol}")
                if not one_image_path:
                    continue

                if force_refresh_vlm:
                    cached_generate_region_findings.clear()

                # 一次 API 调用整张协议图
                one_findings = cached_generate_region_findings(
                    patient_id=str(backend_patient_id),
                    image_path=one_image_path,
                )

                # 附加协议信息
                for finding in safe_list(one_findings):
                    finding = safe_dict(finding)
                    finding["matched_protocol"] = protocol
                    finding["matched_filename"] = path_item.get("filename")
                    finding["matched_rank"] = path_item.get("rank")
                    finding["matched_score"] = path_item.get("score")
                    finding["matched_weight"] = path_item.get("weight")
                    all_region_findings.append(finding)

                progress_bar.progress((idx + 1) / total_images)

            # consistency gate
            vlm_feature_state_dict = build_vlm_feature_state_dict(
                feature_states=feature_states,
                metric_judgements=metric_judgements,
            )
            image_region_findings = check_visual_clinical_consistency(
                region_findings=all_region_findings,
                feature_states=vlm_feature_state_dict,
            )

            st.success("图像侧区域解释生成完成。")

        except Exception as e:
            image_region_error = str(e)
            st.error(f"VLM 推理失败：{e}")

        # -----------------------------
        # 展示结果
        # -----------------------------
        render_image_region_findings_section(
            image_path_for_vlm="；".join([
                f"{x.get('protocol')}: {x.get('image_path')}" for x in image_paths_for_vlm
            ]),
            image_region_findings=image_region_findings,
            image_region_error=image_region_error,
            is_patient_user=is_patient_user,
        )
else:
    st.info("当前未启用图像侧解释。勾选上方选项后，将调用 VLM API 生成区域级视觉解释。")


st.divider()

# -----------------------------
# Representation & RAIR
# -----------------------------
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("ARM 协议阶段贡献（Representation Contribution）")
    protocol_contrib = representation.get("protocol_contribution", {})
    protocol_topk_details = safe_list(representation.get("protocol_topk_details"))

    if not protocol_contrib:
        st.caption("暂无各协议阶段对患者表征的贡献信息。")
    elif isinstance(protocol_contrib, dict) and protocol_contrib.get("available") is False:
        st.caption(protocol_contrib.get("message", "当前患者页暂未接入协议级 attention 贡献明细。"))
    else:
        numeric_items = {
            k: v for k, v in safe_dict(protocol_contrib).items()
            if isinstance(v, (int, float)) and k != "available"
        }
        if numeric_items:
            df_proto = pd.DataFrame(list(numeric_items.items()), columns=["协议阶段", "贡献度"])
            st.bar_chart(df_proto.set_index("协议阶段"), height=220)
            df_proto["贡献度"] = df_proto["贡献度"].apply(lambda x: fmt_number(x, 3))
            st.dataframe(df_proto, use_container_width=True, hide_index=True)
        else:
            st.caption("当前协议贡献结果尚未结构化为可视化数值。")

    if protocol_topk_details and not is_patient_user:
        with st.expander("查看 top-k 关键帧解释"):
            df_topk = pd.DataFrame(protocol_topk_details)
            show_cols = [
                c for c in ["protocol", "rank", "filename", "score", "weight", "image_path", "frame_path"]
                if c in df_topk.columns
            ]
            if not df_topk.empty and show_cols:
                if "score" in df_topk.columns:
                    df_topk["score"] = df_topk["score"].apply(lambda x: fmt_number(x, 3))
                if "weight" in df_topk.columns:
                    df_topk["weight"] = df_topk["weight"].apply(lambda x: fmt_number(x, 3))
                st.dataframe(df_topk[show_cols], use_container_width=True, hide_index=True)
            else:
                st.caption("top-k 关键帧解释结果为空。")
    elif not protocol_topk_details:
        st.caption("暂无 top-k 关键帧解释结果。")

with col_b:
    st.subheader("RAIR 生理反射证据")
    time_series = rair.get("time_series")
    features = rair.get("features", {})

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
            st.caption("RAIR 时间序列存在，但暂时无法可视化。")

    if not features:
        st.caption("暂无 RAIR 患者级特征。")
    elif isinstance(features, dict) and features.get("available") is False:
        st.caption(features.get("message", "暂无 RAIR 患者级特征。"))
    else:
        display_features = {
            "剂量 (ml)": features.get("dose_ml"),
            "剂量是否有效": features.get("dose_valid"),
            "事件编号": features.get("event_id"),
            "事件是否有效": features.get("event_valid"),
            "基线压力": features.get("baseline_pressure"),
            "最低压力": features.get("min_pressure"),
            "松弛幅度": features.get("relaxation_amplitude"),
            "达到最低点时间": features.get("t_min"),
            "是否可恢复": features.get("recovery_possible"),
            "帧数": features.get("n_frames"),
        }
        display_features = {k: v for k, v in display_features.items() if v is not None}
        if display_features:
            df_rair = pd.DataFrame(list(display_features.items()), columns=["特征", "数值"])
            df_rair["数值"] = df_rair["数值"].apply(lambda x: fmt_number(x, 3))
            st.dataframe(df_rair, use_container_width=True, hide_index=True)
        else:
            st.caption("当前患者无可显示的 RAIR 特征。")

    if not is_patient_user:
        with st.expander("调试：查看 RAIR 路径解析"):
            st.json(safe_dict(rair.get("debug")))

st.divider()

# -----------------------------
# Stability & Rome IV
# -----------------------------
col_c, col_d = st.columns(2)

with col_c:
    st.subheader("聚类稳定性细节（随机种子级别）")

    stability_label = stab.get("label", "-")
    stab_confidence = stab.get("confidence")
    switch_rate_val = stab.get("switch_rate")
    is_boundary_val = bool(stab.get("is_boundary", False))

    st.markdown(f"**稳定性标签：** {stability_label}")

    s1, s2, s3 = st.columns(3)
    with s1:
        st.metric("Confidence", fmt_number(stab_confidence, 2))
    with s2:
        st.metric("Switch Rate", fmt_number(switch_rate_val, 2))
    with s3:
        st.metric("边界患者", "是" if is_boundary_val else "否")

    if is_boundary_val:
        st.warning("该患者属于边界患者，说明其聚类归属在不同随机种子下存在一定波动。")
    else:
        st.success("该患者属于稳定患者，说明其聚类归属在不同随机种子下较为一致。")

st.caption(
    f"当前版本采用 confidence ≥ {boundary_threshold} 视为稳定患者，"
    f"confidence < {boundary_threshold} 视为边界患者。"
)

seed_assignments = safe_dict(stab.get("seed_assignments"))
if seed_assignments and not is_patient_user:
    st.markdown("**随机种子分配详情**")
    df_seed = pd.DataFrame(
        list(seed_assignments.items()),
        columns=["随机种子", "分配的 Cluster"]
    ).sort_values("随机种子")
    st.dataframe(df_seed, use_container_width=True, hide_index=True)
elif not seed_assignments and not is_patient_user:
    st.caption("当前未接入逐 seed 标签分配结果。")

with col_d:
    st.subheader("外部临床参考（Rome IV 代理分类）")
    if not rome or rome.get("category") is None:
        st.caption("暂无 Rome IV 代理分类信息。")
    else:
        st.markdown(f"**Rome IV 分类：** {rome.get('category', '-')}")
        st.markdown(f"**推进力：** {rome.get('propulsion', '-') or '-'}")
        st.markdown(f"**协调性：** {rome.get('coordination', '-') or '-'}")

        ratio = rome.get("ratio_msp_mrp")
        if ratio is not None:
            st.write(f"**MSP/MRP 比值：** {fmt_number(ratio, 3)}")

        if rome.get("proxy_type"):
            st.caption(f"代理类型：{rome.get('proxy_type')}")

        rules = safe_list(rome.get("rules"))
        if rules:
            st.markdown("**判定依据**")
            for rule in rules:
                st.write(f"- {rule}")

st.divider()

# -----------------------------
# Group statistics
# -----------------------------
version = current_version.get("display_name", selected_version)
summary = ""
suggestion = ""

if group_stats:
    version = group_stats.get("version", version)
    summary = group_stats.get("summary", "")
    suggestion = group_stats.get("suggestion", "")

if not summary:
    summary = (
        f"当前患者页展示的是 {current_version.get('short_name', selected_version)} "
        f"版本下的患者分型结果。"
    )

if not suggestion:
    suggestion = "请结合患者临床指标、RAIR 反射证据及 Rome IV proxy 信息进行科研解释。"

if not is_patient_user:
    st.subheader("群体统计参考")
    st.info(f"版本：{version}\n\n{summary}\n\n{suggestion}")
    st.page_link(
        "pages/4_Statistics_View.py",
        label="打开 Statistics View 查看完整统计结果",
        icon="📊"
    )
    st.divider()

# -----------------------------
# LLM analysis
# -----------------------------
st.subheader("🤖 AI 智能分析")

llm_status = get_llm_runtime_status()

st.caption(
    "LLM_MODE: "
    f"use_real_api={llm_status.get('use_real_api')}, "
    f"enable_raw={llm_status.get('enable_raw')}, "
    f"provider={llm_status.get('provider')}, "
    f"model={llm_status.get('model')}, "
    f"env={llm_status.get('env_path')}"
)

kg_paths_for_llm = []

if GRAPH_FEATURE_AVAILABLE:
    try:
        kg_for_llm = load_patient_graph(backend_patient_id)
        kg_paths_for_llm = safe_list(kg_for_llm.get("paths"))
    except Exception:
        kg_paths_for_llm = []

# ============================================================
# LLM 优先使用当前版本 CSV 结果；后端结果作为补充
# 这样 Streamlit Cloud 后端 patient 为空时，LLM 仍可解释当前版本分型和医院指标
# ============================================================

backend_metric_judgements = safe_list(
    patient.get("metric_judgements")
    or patient.get("metric_judgments")
    or []
)

backend_feature_states = safe_list(
    patient.get("feature_states")
    or []
)

# 当前版本 CSV 中提取出的医院参考范围判定 / feature states 优先
llm_metric_judgements = safe_list(metric_judgements) or backend_metric_judgements
llm_feature_states = safe_list(feature_states) or backend_feature_states

# ai / stab 已经在前面被 version_patient_result 覆盖过，所以这里必须用 ai / stab
# 不能继续用 backend_ai / backend_stab
llm_context = build_llm_context(
    patient_id=backend_patient_id or patient_id,
    ai=ai,
    stability=stab,
    metric_judgements=llm_metric_judgements,
    feature_states=llm_feature_states,
    rair=rair,
    rome=rome,
    rag=rag,
    kg_paths=kg_paths_for_llm,
)

with st.expander("调试：查看最终 LLM 输入来源"):
    st.write("ai:", ai)
    st.write("stab:", stab)
    st.write("metric_judgements count:", len(metric_judgements))
    st.write("feature_states count:", len(feature_states))
    st.write("backend_metric_judgements count:", len(backend_metric_judgements))
    st.write("backend_feature_states count:", len(backend_feature_states))
    st.json(llm_context)

if not isinstance(llm_context, dict):
    llm_context = {}

image_region_split = split_image_region_findings_for_llm(image_region_findings)

llm_context["image_region_findings"] = image_region_split["usable"]
llm_context["image_region_uncertain_findings"] = image_region_split["uncertain_or_conflict"]
llm_context["image_region_usage_rules"] = [
    "图像侧区域解释仅作为辅助证据，不参与无监督分型。",
    "不得根据图像侧区域解释重新判断患者 cluster。",
    "不得根据图像侧区域解释输出临床诊断或治疗建议。",
    "consistency_status 为 conflict、uncertain 或 weak_visual_evidence 的结果只能进入不确定性说明。",
    "最终解释应以无监督分型、医院参考范围判定、RAIR / Rome IV proxy、KG 路径和 RAG 文献证据为主。",
]

generated_report = generate_llm_report(llm_context)

tab_generated, tab_backend, tab_debug = st.tabs(
    ["结构化科研解释报告", "后端原始 AI 分析", "调试输入"]
)

with tab_generated:
    st.caption(
        "该报告由当前页面结构化结果生成。LLM/规则解释模块不参与患者分型，"
        "仅用于科研解释与系统展示。"
    )
    st.markdown(generated_report)

with tab_backend:
    st.markdown("**分析摘要**")
    st.write(llm_analysis.get("summary", "暂无分析摘要。"))

    st.markdown("**关键发现**")
    key_findings = safe_list(llm_analysis.get("key_findings"))
    if key_findings:
        for finding in key_findings:
            st.write(f"- {finding}")
    else:
        st.caption("暂无关键发现。")

    st.markdown("**临床意义**")
    st.write(llm_analysis.get("clinical_significance", "暂无临床意义说明。"))

    st.markdown("**建议**")
    recommendations = safe_list(llm_analysis.get("recommendations"))
    if recommendations:
        for rec in recommendations:
            st.write(f"- {rec}")
    else:
        st.caption("暂无建议。")

with tab_debug:
    if not is_patient_user:
        st.json(llm_context)
    else:
        st.caption("患者角色不展示调试输入。")

st.divider()

# -----------------------------
# Knowledge Graph
# -----------------------------
st.subheader("🕸️ 知识图谱解释")

if not GRAPH_FEATURE_AVAILABLE:
    st.info("当前环境未安装 Neo4j / pyvis 相关依赖，知识图谱模块暂不可用。其他患者分析功能可正常使用。")
    if GRAPH_IMPORT_ERROR:
        st.caption(f"导入信息：{GRAPH_IMPORT_ERROR}")
    st.divider()
else:
    kg_btn_col1, kg_btn_col2 = st.columns([1, 4])
    with kg_btn_col1:
        refresh_graph_btn = st.button("生成/刷新知识图谱", use_container_width=True)

    graph_error = None
    if refresh_graph_btn:
        with st.spinner("正在构建知识图谱..."):
            graph_error = sync_patient_graph(backend_patient_id, patient)
        if graph_error:
            st.error(f"知识图谱更新失败：{graph_error}")
        else:
            st.success("知识图谱已更新。")

    try:
        kg = load_patient_graph(backend_patient_id)
    except Exception as e:
        kg = {"nodes": [], "edges": [], "paths": []}
        st.caption(f"知识图谱暂不可用：{e}")

    if not kg.get("nodes"):
        st.caption("当前患者暂无知识图谱数据。可点击上方按钮生成。")
    else:
        st.markdown("**关键解释路径**")
        paths = safe_list(kg.get("paths"))
        if paths:
            for idx, path in enumerate(paths, 1):
                st.write(f"{idx}. " + " → ".join([str(x) for x in path if x]))
        else:
            st.caption("暂无解释路径。")

        kg_summary_col1, kg_summary_col2, kg_summary_col3 = st.columns(3)
        with kg_summary_col1:
            st.metric("节点数", len(safe_list(kg.get("nodes"))))
        with kg_summary_col2:
            st.metric("关系数", len(safe_list(kg.get("edges"))))
        with kg_summary_col3:
            st.metric("解释路径数", len(paths))

        st.markdown("**图谱可视化（已自动精简显示）**")
        render_knowledge_graph_pyvis(
            safe_list(kg.get("nodes")),
            safe_list(kg.get("edges")),
            height=760,
        )

        with st.expander("查看图谱节点"):
            nodes_df = pd.DataFrame(safe_list(kg.get("nodes")))
            if not nodes_df.empty:
                st.dataframe(nodes_df, use_container_width=True, hide_index=True)
            else:
                st.caption("暂无节点。")

        with st.expander("查看图谱关系"):
            edges_df = pd.DataFrame(safe_list(kg.get("edges")))
            if not edges_df.empty:
                st.dataframe(edges_df, use_container_width=True, hide_index=True)
            else:
                st.caption("暂无关系。")

    st.divider()

# -----------------------------
# RAG section
# -----------------------------
if not is_patient_user:
    st.subheader("📚 文献检索解释（RAG）")

    rag_input_features = safe_dict(rag.get("input_features"))
    rag_chunks = safe_list(rag.get("retrieved_chunks"))
    rag_explanation = safe_dict(rag.get("explanation"))

    left_rag, right_rag = st.columns(2)

    with left_rag:
        st.markdown("**检索输入特征**")
        if rag_input_features:
            df_rag_input = pd.DataFrame(
                list(rag_input_features.items()),
                columns=["特征", "取值"]
            )
            st.dataframe(df_rag_input, use_container_width=True, hide_index=True)
        else:
            st.caption("暂无 RAG 输入特征。")

    with right_rag:
        st.markdown("**解释摘要**")
        st.write(rag_explanation.get("summary", "暂无"))
        st.markdown("**解释说明**")
        st.write(rag_explanation.get("interpretation", "暂无"))
        st.markdown("**不确定性**")
        st.info(rag_explanation.get("uncertainty", "暂无"))

    st.markdown("**召回证据明细**")
    if rag_chunks:
        for i, chunk in enumerate(rag_chunks, 1):
            chunk = safe_dict(chunk)
            chunk_id = chunk.get("chunk_id", "")
            score = chunk.get("score", 0)
            title = chunk.get("title", "")
            source = chunk.get("source", "")
            matched_terms = chunk.get("matched_terms", [])
            matched_tags = chunk.get("matched_tags", [])
            chunk_text = chunk.get("chunk_text", "")

            exp_title = f"{i}. {chunk_id or '未知Chunk'} | score={fmt_number(score, 3)}"
            with st.expander(exp_title):
                st.write(f"**标题：** {title or '-'}")
                st.write(f"**来源：** {source or '-'}")
                st.write(f"**匹配词：** {matched_terms if matched_terms else '-'}")
                st.write(f"**匹配标签：** {matched_tags if matched_tags else '-'}")
                st.write("**证据内容：**")
                st.write(chunk_text or "暂无内容。")
    else:
        st.warning("当前没有召回到文献证据。请检查知识库字段、标签映射或检索规则。")

    st.divider()

    st.subheader("📚 专家知识推荐")
    if not rag_recommendations:
        st.caption("暂无专家知识推荐。")
    else:
        for i, rec in enumerate(rag_recommendations, 1):
            rec = safe_dict(rec)
            title = rec.get("title", f"推荐 {i}")
            score = rec.get("score")
            content = rec.get("text", "")
            source = rec.get("source", "未知来源")
            chunk_id = rec.get("chunk_id", "")

            header = f"{i}. {title}"
            if score is not None:
                try:
                    header += f"（score: {float(score):.2f}）"
                except Exception:
                    pass

            with st.expander(header):
                if chunk_id:
                    st.write(f"**Chunk ID：** {chunk_id}")
                st.write(content or "暂无内容。")
                st.caption(f"来源: {source}")

    st.divider()

    with st.expander("调试：查看原始 RAG 输出"):
        st.json(rag)

st.caption("⚠️ 本系统仅用于科研与功能表型分析，不用于临床诊断或治疗决策。")