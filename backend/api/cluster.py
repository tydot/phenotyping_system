"""
backend/api/cluster.py

Cluster-level API
基于 SQLite 中的真实共识分型结果与临床联合表，生成集群视图
并接入 cluster-level RAG 解释
"""

from typing import Dict, Any, List
import math
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.auth.auth_service import require_login

require_login()

from backend.db.database import get_conn

from backend.db.query_rag_kb import load_cluster_kb
from backend.rag.retriever import retrieve_top_chunks_for_cluster
from backend.rag.generator import generate_cluster_explanation


def _safe_float(x):
    if x is None:
        return None
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def _safe_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if x in (1, "1", "true", "True", "TRUE", "yes", "Yes"):
        return True
    return False


def _median(values: List[Any]):
    vals = [_safe_float(v) for v in values]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    vals = sorted(vals)
    n = len(vals)
    mid = n // 2
    if n % 2 == 1:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def _rate(cond_list: List[bool]) -> float:
    if not cond_list:
        return 0.0
    return sum(bool(x) for x in cond_list) / len(cond_list)


def _load_cluster_rows(cluster_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT
            pc.patient_id,
            pc.consensus_cluster,
            pc.confidence,
            pc.switch_rate,
            pcons.is_boundary,

            pc.sex,
            pc.age,
            pc.main_symptom,

            pc.resting_pressure,
            pc.msp,
            pc.squeeze_duration,
            pc.defecatory_rectal_pressure,

            pc.first_sensation,
            pc.desire_to_defecate,
            pc.urgency_threshold,
            pc.max_tolerable_volume,
            pc.rair_min_volume,
            pc.anal_length
        FROM patient_clinical pc
        LEFT JOIN patient_consensus pcons
            ON pc.patient_id = pcons.patient_id
        WHERE pc.consensus_cluster = ?
    """, (_safe_int(cluster_id),))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _build_profile(rows):
    median_profile = {
        "肛门括约肌静息压 (mmHg)": _median([r.get("resting_pressure") for r in rows]),
        "最大缩榨压 MSP (mmHg)": _median([r.get("msp") for r in rows]),
        "缩肛持续时间 (s)": _median([r.get("squeeze_duration") for r in rows]),
        "排便时直肠压力 (mmHg)": _median([r.get("defecatory_rectal_pressure") for r in rows]),
        "肛门括约肌长度 (cm)": _median([r.get("anal_length") for r in rows]),
        "最大容量感觉阈值 (ml)": _median([r.get("max_tolerable_volume") for r in rows]),
    }

    abnormality_rate = {
        "低静息压比例": _rate([
            (_safe_float(r.get("resting_pressure")) is not None and _safe_float(r.get("resting_pressure")) < 40)
            for r in rows
        ]),
        "低 MSP 比例": _rate([
            (_safe_float(r.get("msp")) is not None and _safe_float(r.get("msp")) < 120)
            for r in rows
        ]),
        "排便推动不足比例": _rate([
            (_safe_float(r.get("defecatory_rectal_pressure")) is not None and _safe_float(r.get("defecatory_rectal_pressure")) < 45)
            for r in rows
        ]),
        "缩肛持续时间异常比例": _rate([
            (_safe_float(r.get("squeeze_duration")) is not None and _safe_float(r.get("squeeze_duration")) < 3)
            for r in rows
        ]),
        "感觉阈值升高比例": _rate([
            (_safe_float(r.get("max_tolerable_volume")) is not None and _safe_float(r.get("max_tolerable_volume")) > 250)
            for r in rows
        ]),
    }

    # RAIR 相关统计
    # rair_min_volume 表示最小诱发剂量：
    # - 非空：说明 RAIR 可被诱发，且记录到了最小诱发剂量
    # - 中位数：反映 cluster 内 RAIR 最小诱发剂量分布水平
    rair_stats = {
        "RAIR 可诱发比例": _rate([
            (_safe_float(r.get("rair_min_volume")) is not None)
            for r in rows
        ]),
        "RAIR 最小诱发剂量中位数 (ml)": _median([
            r.get("rair_min_volume") for r in rows
        ]),
        "RAIR 数据缺失比例": _rate([
            (_safe_float(r.get("rair_min_volume")) is None)
            for r in rows
        ]),
    }

    return median_profile, abnormality_rate, rair_stats


def _build_cluster_description(
    cluster_id: int,
    median_profile: Dict[str, Any],
    abnormality_rate: Dict[str, Any],
    rair_stats: Dict[str, Any],
) -> str:
    rp = median_profile.get("肛门括约肌静息压 (mmHg)")
    msp = median_profile.get("最大缩榨压 MSP (mmHg)")
    drp = median_profile.get("排便时直肠压力 (mmHg)")
    max_tv = median_profile.get("最大容量感觉阈值 (ml)")
    rair_rate = rair_stats.get("RAIR 可诱发比例")
    rair_min_median = rair_stats.get("RAIR 最小诱发剂量中位数 (ml)")

    parts = [f"Cluster {cluster_id} 的患者在关键 ARM 功能指标上呈现相对一致的表型特征。"]

    if rp is not None:
        if rp < 40:
            parts.append("该簇整体静息压偏低，提示括约肌基础张力相对不足。")
        elif rp > 60:
            parts.append("该簇整体静息压偏高，提示括约肌基础张力较强。")
        else:
            parts.append("该簇静息压整体处于中间水平。")

    if msp is not None:
        if msp < 120:
            parts.append("最大缩榨压整体偏低，提示主动收缩能力相对减弱。")
        else:
            parts.append("最大缩榨压整体尚可，提示主动收缩能力相对保留。")

    if drp is not None:
        if drp < 45:
            parts.append("排便时直肠压力整体偏低，提示推进力不足倾向。")
        else:
            parts.append("排便时直肠压力相对较好，提示推进力相对保留。")

    if max_tv is not None:
        if max_tv > 250:
            parts.append("最大可耐受容量偏高，提示直肠感觉阈值升高或感觉减退倾向。")
        elif max_tv < 120:
            parts.append("最大可耐受容量偏低，提示直肠感觉较敏感。")

    if rair_rate is not None:
        parts.append(f"该簇 RAIR 可诱发比例为 {rair_rate:.1%}。")

    if rair_min_median is not None:
        parts.append(f"在可诱发病例中，RAIR 最小诱发剂量中位数为 {rair_min_median:.2f} ml。")

    return "".join(parts)


def build_cluster_rag_features(
    cluster_id: int,
    median_profile: Dict[str, Any],
    abnormality_rate: Dict[str, Any],
):
    summary_features: List[str] = []

    rp = _safe_float(median_profile.get("肛门括约肌静息压 (mmHg)"))
    msp = _safe_float(median_profile.get("最大缩榨压 MSP (mmHg)"))
    drp = _safe_float(median_profile.get("排便时直肠压力 (mmHg)"))
    squeeze_duration = _safe_float(median_profile.get("缩肛持续时间 (s)"))
    max_tv = _safe_float(median_profile.get("最大容量感觉阈值 (ml)"))

    low_rest_rate = _safe_float(abnormality_rate.get("低静息压比例")) or 0.0
    low_msp_rate = _safe_float(abnormality_rate.get("低 MSP 比例")) or 0.0
    poor_prop_rate = _safe_float(abnormality_rate.get("排便推动不足比例")) or 0.0
    short_squeeze_rate = _safe_float(abnormality_rate.get("缩肛持续时间异常比例")) or 0.0
    high_threshold_rate = _safe_float(abnormality_rate.get("感觉阈值升高比例")) or 0.0

    if rp is not None:
        if rp < 40:
            summary_features.extend(["low resting pressure", "low resting tone"])
        elif rp > 60:
            summary_features.extend(["high resting pressure", "high anal tone"])

    if msp is not None:
        if msp < 120:
            summary_features.extend(["low squeeze pressure", "weak squeeze", "sphincter weakness"])
        elif msp > 180:
            summary_features.append("high squeeze pressure")

    if drp is not None:
        if drp < 45:
            summary_features.extend(["poor propulsion", "low rectal propulsive force"])
        elif drp > 80:
            summary_features.append("high rectal propulsive force")

    if squeeze_duration is not None and squeeze_duration < 3:
        summary_features.extend(["short squeeze duration", "impaired voluntary contraction"])

    if max_tv is not None:
        if max_tv > 250:
            summary_features.extend(["hyposensitivity", "high sensory threshold", "sensory dysfunction"])
        elif max_tv < 120:
            summary_features.extend(["hypersensitivity", "low sensory threshold"])

    if poor_prop_rate >= 0.5:
        summary_features.extend(["cluster poor propulsion", "phenotype poor propulsion"])

    if low_msp_rate >= 0.5:
        summary_features.extend(["cluster weak squeeze", "phenotype sphincter weakness"])

    if low_rest_rate >= 0.5:
        summary_features.extend(["cluster low resting tone", "phenotype low resting pressure"])

    if high_threshold_rate >= 0.4:
        summary_features.extend(["cluster hyposensitivity", "rectal sensory dysfunction"])

    if short_squeeze_rate >= 0.4:
        summary_features.extend(["cluster impaired contractility", "reduced squeeze endurance"])

    if poor_prop_rate >= 0.5 and low_msp_rate >= 0.5:
        summary_features.extend(["dyssynergia", "defecatory dysfunction", "biofeedback candidate"])
    elif poor_prop_rate >= 0.5:
        summary_features.extend(["defecatory dysfunction", "biofeedback relevance"])

    deduped = []
    seen = set()
    for x in summary_features:
        key = x.strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(x)

    return {
        "page_type": "cluster",
        "cluster_id": cluster_id,
        "summary_features": deduped,
    }


def get_cluster_rag_explanation(cluster_features: dict, top_k: int = 5):
    try:
        df = load_cluster_kb()
        chunks = retrieve_top_chunks_for_cluster(
            df=df,
            summary_features=cluster_features.get("summary_features", []),
            top_k=top_k,
        )

        cluster_input = {
            "page_type": "cluster",
            "cluster_id": cluster_features.get("cluster_id"),
            "summary_features": cluster_features.get("summary_features", []),
        }

        explanation = generate_cluster_explanation(
            cluster_input=cluster_input,
            retrieved_chunks=chunks,
        )

        return {
            "input_features": cluster_features,
            "retrieved_chunks": chunks,
            "explanation": explanation,
        }
    except Exception as e:
        return {
            "input_features": cluster_features,
            "retrieved_chunks": [],
            "explanation": {
                "summary": "Cluster 知识库解释模块暂时不可用。",
                "interpretation": f"RAG 模块调用失败：{e}",
                "uncertainty": "请检查知识库 Excel 路径、cluster 检索函数、以及 cluster generator 是否已配置。",
                "evidence": [],
            },
        }


def get_cluster_view(cluster_id: int) -> Dict[str, Any]:
    cluster_id = _safe_int(cluster_id)
    rows = _load_cluster_rows(cluster_id)

    if not rows:
        return {
            "cluster_id": cluster_id,
            "size": 0,
            "stable_ratio": 0.0,
            "median_profile": {},
            "abnormality_rate": {},
            "rair_stats": {},
            "phenotype_description": "暂无该 Cluster 的数据。",
            "llm_analysis": {
                "summary": "暂无该 Cluster 的数据。",
                "key_findings": [],
                "clinical_significance": "暂无可用于 cluster-level 解释的数据。",
                "recommendations": [],
            },
            "rag": {
                "input_features": {},
                "retrieved_chunks": [],
                "explanation": {
                    "summary": "",
                    "interpretation": "",
                    "uncertainty": "",
                    "evidence": [],
                },
            },
            "rag_recommendations": [],
            "rag_evidence": [],
        }

    size = len(rows)
    stable_count = sum(1 for r in rows if not _safe_bool(r.get("is_boundary")))
    stable_ratio = stable_count / size if size > 0 else 0.0

    median_profile, abnormality_rate, rair_stats = _build_profile(rows)
    phenotype_description = _build_cluster_description(
        cluster_id=cluster_id,
        median_profile=median_profile,
        abnormality_rate=abnormality_rate,
        rair_stats=rair_stats,
    )

    cluster_rag_features = build_cluster_rag_features(
        cluster_id=cluster_id,
        median_profile=median_profile,
        abnormality_rate=abnormality_rate,
    )

    rag_result = get_cluster_rag_explanation(cluster_rag_features, top_k=5)
    rag_explanation = rag_result.get("explanation", {})
    rag_chunks = rag_result.get("retrieved_chunks", [])

    rag_recommendations = []
    for item in rag_explanation.get("evidence", []):
        rag_recommendations.append({
            "chunk_id": item.get("chunk_id", ""),
            "title": item.get("title", ""),
            "source": item.get("source", ""),
            "score": item.get("score", 0),
            "text": item.get("text", ""),
        })

    key_findings = [
        f"Cluster 大小：{size}",
        f"稳定患者比例：{stable_ratio:.2f}",
    ]

    rp = median_profile.get("肛门括约肌静息压 (mmHg)")
    msp = median_profile.get("最大缩榨压 MSP (mmHg)")
    drp = median_profile.get("排便时直肠压力 (mmHg)")
    max_tv = median_profile.get("最大容量感觉阈值 (ml)")
    rair_rate = rair_stats.get("RAIR 可诱发比例")
    rair_min_median = rair_stats.get("RAIR 最小诱发剂量中位数 (ml)")

    if rp is not None:
        key_findings.append(f"中位静息压：{rp:.2f} mmHg")
    if msp is not None:
        key_findings.append(f"中位 MSP：{msp:.2f} mmHg")
    if drp is not None:
        key_findings.append(f"中位排便直肠压力：{drp:.2f} mmHg")
    if max_tv is not None:
        key_findings.append(f"中位最大容量感觉阈值：{max_tv:.2f} ml")
    if rair_rate is not None:
        key_findings.append(f"RAIR 可诱发比例：{rair_rate:.1%}")
    if rair_min_median is not None:
        key_findings.append(f"RAIR 最小诱发剂量中位数：{rair_min_median:.2f} ml")

    if rag_explanation.get("summary"):
        key_findings.append(f"知识库解释摘要：{rag_explanation['summary']}")

    recommendations = [
        "该页面展示的是 cluster-level 画像与知识支持性解释，不替代个体患者诊断判断。",
        "异常比例与 RAIR 统计已分开展示，避免将 RAIR 可诱发性误解为一般异常率。",
    ]
    if rag_explanation.get("uncertainty"):
        recommendations.append(f"解释不确定性：{rag_explanation['uncertainty']}")

    clinical_significance = phenotype_description
    if rag_explanation.get("interpretation"):
        clinical_significance += f" 基于文献知识库的 cluster-level 解释提示：{rag_explanation['interpretation']}"

    return {
        "cluster_id": cluster_id,
        "size": size,
        "stable_ratio": stable_ratio,
        "median_profile": median_profile,
        "abnormality_rate": abnormality_rate,
        "rair_stats": rair_stats,
        "phenotype_description": phenotype_description,
        "llm_analysis": {
            "summary": (
                rag_explanation.get("summary")
                or phenotype_description
            ),
            "key_findings": key_findings,
            "clinical_significance": clinical_significance,
            "recommendations": recommendations,
        },
        "rag": {
            "input_features": cluster_rag_features,
            "retrieved_chunks": rag_chunks,
            "explanation": rag_explanation,
        },
        "rag_recommendations": rag_recommendations,
        "rag_evidence": rag_chunks,
    }