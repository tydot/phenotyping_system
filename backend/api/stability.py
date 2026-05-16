
"""
backend/api/stability.py

Stability-level API
基于 SQLite 中的真实共识分型结果进行稳定性分析
"""
import sys
from pathlib import Path
from typing import Dict, Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.auth.auth_service import require_login

require_login()
from backend.db.query_stability import get_all_patient_consensus, get_boundary_patients


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def get_stability_view() -> Dict[str, Any]:
    rows = get_all_patient_consensus() or []

    if not rows:
        return {
            "cohort_stability": {
                "stable": 0.0,
                "boundary": 0.0,
            },
            "cluster_stability": {},
            "confidence_distribution": [],
            "boundary_patients": [],
            "llm_interpretation": {
                "overall_assessment": "暂无稳定性数据。",
                "cluster_analysis": "暂无集群稳定性结果。",
                "confidence_analysis": "暂无置信度分布结果。",
                "recommendations": [],
            },
        }

    total_patients = len(rows)
    stable_count = sum(1 for r in rows if not bool(r.get("is_boundary")))
    boundary_count = sum(1 for r in rows if bool(r.get("is_boundary")))

    cohort_stability = {
        "stable": stable_count / total_patients,
        "boundary": boundary_count / total_patients,
    }

    cluster_groups = {}
    for r in rows:
        c = int(r.get("consensus_cluster", -1))
        cluster_groups.setdefault(c, []).append(r)

    cluster_stability = {}
    for c, group in sorted(cluster_groups.items(), key=lambda x: x[0]):
        stable_in_cluster = sum(1 for r in group if not bool(r.get("is_boundary")))
        cluster_stability[c] = stable_in_cluster / len(group) if group else 0.0

    confidence_distribution = [_safe_float(r.get("confidence")) for r in rows]
    boundary_patients = get_boundary_patients() or []

    llm_interpretation = {
        "overall_assessment": (
            f"当前共识分型共纳入 {total_patients} 名患者，"
            f"其中稳定患者 {stable_count} 名，边界患者 {boundary_count} 名。"
        ),
        "cluster_analysis": "各 cluster 的稳定性比例反映了该分型结构在多随机种子下的一致性。",
        "confidence_analysis": (
            "confidence 表示患者在多次聚类中被分配到最终共识 cluster 的比例；"
            "confidence < 0.8 定义为边界患者，confidence >= 0.8 定义为稳定患者。"
        ),
        "recommendations": [
            "优先基于稳定患者开展临床统计分析。",
            "边界患者可单独复核，以评估分型边界区域的异质性。",
        ],
    }

    return {
        "cohort_stability": cohort_stability,
        "cluster_stability": cluster_stability,
        "confidence_distribution": confidence_distribution,
        "boundary_patients": boundary_patients,
        "llm_interpretation": llm_interpretation,
    }
