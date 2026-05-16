import sys
from pathlib import Path
from typing import Any, Dict, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.auth.auth_service import require_login

require_login()

from backend.db.query_patient import get_patient_consensus, get_patient_clinical
from backend.db.query_seed_assignments import get_patient_seed_assignments
from backend.db.query_protocol_contribution import (
    get_patient_protocol_contribution,
    get_patient_protocol_topk_details,
)
from backend.db.query_rair import get_patient_rair_features, get_patient_rair_time_series

from backend.db.query_rag_kb import load_patient_kb
from backend.rag.retriever import retrieve_top_chunks_for_patient
from backend.rag.generator import generate_patient_explanation
from backend.clinical.feature_rules import build_patient_feature_states, normalize_gender


def normalize_pid(x):
    if x is None:
        return None
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s or None


def _safe_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def resolve_gender(clinical: Dict[str, Any], consensus: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    显式解析性别来源，避免 silently fallback。
    """
    clinical_gender = clinical.get("gender") if isinstance(clinical, dict) else None
    consensus_gender = consensus.get("gender") if isinstance(consensus, dict) else None

    if clinical_gender not in [None, ""]:
        resolved = normalize_gender(clinical_gender)
        return resolved, {
            "resolved_gender": resolved,
            "source": "clinical",
            "raw_value": clinical_gender,
            "is_defaulted": False,
        }

    if consensus_gender not in [None, ""]:
        resolved = normalize_gender(consensus_gender)
        return resolved, {
            "resolved_gender": resolved,
            "source": "consensus",
            "raw_value": consensus_gender,
            "is_defaulted": False,
        }

    resolved = "female"
    return resolved, {
        "resolved_gender": resolved,
        "source": "default",
        "raw_value": None,
        "is_defaulted": True,
        "note": "clinical 与 consensus 均未提供 gender，当前按 female 参考值回退。",
    }


def build_rome_iv_proxy(clinical: dict, gender: str = "female"):
    """
    基于统一 clinical.feature_rules 的简化 Rome IV proxy。
    不再额外使用独立硬编码阈值，只消费统一特征状态。

    说明：
    - 这是研究/解释用途的 proxy，不是正式临床诊断器。
    - 目前主要根据“推进力 + 收缩/感觉/RAIR 等异常特征”生成辅助分类描述。
    """
    if not clinical:
        return {
            "category": None,
            "propulsion": None,
            "coordination": None,
            "rules": [],
            "proxy_type": "unavailable",
            "based_on_unified_feature_rules": False,
        }

    features = build_patient_feature_states(
        clinical=clinical or {},
        gender=normalize_gender(gender),
    )

    rules = []

    rp = _safe_float(clinical.get("resting_pressure"))
    msp = _safe_float(clinical.get("msp"))
    squeeze_duration = _safe_float(clinical.get("squeeze_duration"))
    drp = _safe_float(clinical.get("defecatory_rectal_pressure"))
    first_sensation = _safe_float(clinical.get("first_sensation"))
    urge_threshold = _safe_float(clinical.get("desire_to_defecate"))
    distress_threshold = _safe_float(clinical.get("urgency_threshold"))
    max_tolerable_volume = _safe_float(clinical.get("max_tolerable_volume"))
    rair_min_volume = _safe_float(clinical.get("rair_min_volume"))
    anal_length = _safe_float(clinical.get("anal_length"))

    # 1) 推进力：完全依据统一规则
    drp_state = features.get("defecatory_rectal_pressure")
    propulsion = None

    if drp_state == "low":
        propulsion = "推进力不足"
        if drp is not None:
            rules.append(f"排便时直肠压力 = {drp:.2f}，统一规则判定为 low")
    elif drp_state in {"normal", "high"}:
        propulsion = "推进力保留"
        if drp is not None:
            rules.append(f"排便时直肠压力 = {drp:.2f}，统一规则判定为 {drp_state}")

    # 2) 协调性：不再使用旧 ratio / <3s 硬编码
    #    改为根据统一规则下的异常特征做解释性 proxy
    coordination_flags = []

    if features.get("resting_pressure") == "high":
        coordination_flags.append("静息压偏高")
        if rp is not None:
            rules.append(f"静息压 = {rp:.2f}，统一规则判定为 high")

    if features.get("msp") == "low":
        coordination_flags.append("缩榨压偏低")
        if msp is not None:
            rules.append(f"最大缩榨压 = {msp:.2f}，统一规则判定为 low")

    if features.get("squeeze_duration") == "low":
        coordination_flags.append("缩肛持续时间偏短")
        if squeeze_duration is not None:
            rules.append(f"缩肛持续时间 = {squeeze_duration:.2f}，统一规则判定为 low")

    if features.get("rair") == "abnormal":
        coordination_flags.append("RAIR 异常")
        if rair_min_volume is not None:
            rules.append(f"RAIR 诱发最小容积 = {rair_min_volume:.2f}，统一规则判定为 abnormal")

    if features.get("rectal_sensation") == "high_threshold":
        coordination_flags.append("直肠感觉阈值升高")
        vals = []
        if first_sensation is not None:
            vals.append(f"初始感觉={first_sensation:.2f}")
        if urge_threshold is not None:
            vals.append(f"初始便意={urge_threshold:.2f}")
        if distress_threshold is not None:
            vals.append(f"窘迫感阈值={distress_threshold:.2f}")
        if max_tolerable_volume is not None:
            vals.append(f"最大耐受量={max_tolerable_volume:.2f}")
        if vals:
            rules.append("感觉功能相关指标提示阈值升高：" + "，".join(vals))

    elif features.get("rectal_sensation") == "low_threshold":
        coordination_flags.append("直肠感觉阈值降低")
        vals = []
        if first_sensation is not None:
            vals.append(f"初始感觉={first_sensation:.2f}")
        if urge_threshold is not None:
            vals.append(f"初始便意={urge_threshold:.2f}")
        if distress_threshold is not None:
            vals.append(f"窘迫感阈值={distress_threshold:.2f}")
        if max_tolerable_volume is not None:
            vals.append(f"最大耐受量={max_tolerable_volume:.2f}")
        if vals:
            rules.append("感觉功能相关指标提示阈值降低：" + "，".join(vals))

    if features.get("anal_length") == "low":
        coordination_flags.append("肛管长度偏短")
        if anal_length is not None:
            rules.append(f"肛门括约肌长度 = {anal_length:.2f}，统一规则判定为 low")
    elif features.get("anal_length") == "high":
        coordination_flags.append("肛管长度偏长")
        if anal_length is not None:
            rules.append(f"肛门括约肌长度 = {anal_length:.2f}，统一规则判定为 high")

    coordination = "疑似协调/排便功能异常" if coordination_flags else "未见明显协调异常"

    # 3) 生成 proxy category
    if propulsion == "推进力不足":
        category = "Poor Propulsion"
    elif coordination_flags:
        category = "Dyssynergic-like"
    elif propulsion == "推进力保留":
        category = "Normal-like"
    else:
        category = None

    return {
        "category": category,
        "propulsion": propulsion,
        "coordination": coordination,
        "coordination_flags": coordination_flags,
        "ratio_msp_mrp": None,  # 明确停用旧 ratio proxy
        "rules": rules,
        "proxy_type": "simplified_proxy",
        "based_on_unified_feature_rules": True,
    }


def build_patient_rag_features(clinical: dict, gender: str = "female"):
    """
    统一复用 clinical.feature_rules 中的规则。
    """
    return build_patient_feature_states(
        clinical=clinical or {},
        gender=normalize_gender(gender),
    )


def get_patient_rag_explanation(patient_features: dict, top_k: int = 5):
    try:
        df = load_patient_kb()
        chunks = retrieve_top_chunks_for_patient(
            df=df,
            features=patient_features,
            top_k=top_k,
        )

        patient_input = {
            "page_type": "patient",
            "features": patient_features,
        }

        explanation = generate_patient_explanation(
            patient_input=patient_input,
            retrieved_chunks=chunks,
        )

        return {
            "input_features": patient_features,
            "retrieved_chunks": chunks,
            "explanation": explanation,
        }
    except Exception as e:
        return {
            "input_features": patient_features,
            "retrieved_chunks": [],
            "explanation": {
                "summary": "知识库解释模块暂时不可用。",
                "interpretation": f"RAG 模块调用失败：{e}",
                "uncertainty": "请检查知识库 Excel 路径、列名映射以及 backend/rag 模块是否已创建。",
                "evidence": [],
            },
        }


def get_patient_view(patient_id: str):
    patient_id = normalize_pid(patient_id)
    if not patient_id:
        return None

    consensus = get_patient_consensus(patient_id)
    if consensus is None:
        return None

    clinical = get_patient_clinical(patient_id) or {}

    gender, gender_meta = resolve_gender(clinical, consensus)

    cluster = int(consensus.get("consensus_cluster", -1)) if consensus.get("consensus_cluster") is not None else None
    confidence = _safe_float(consensus.get("confidence")) or 0.0
    switch_rate = _safe_float(consensus.get("switch_rate")) or 0.0
    is_boundary = bool(consensus.get("is_boundary"))

    core_metrics = {}
    descriptive_metrics = {}

    mapping_core = [
        ("resting_pressure", "肛门括约肌静息压 (mmHg)"),
        ("msp", "最大缩榨压 MSP (mmHg)"),
        ("squeeze_duration", "缩肛持续时间 (s)"),
        ("defecatory_rectal_pressure", "排便时直肠压力 (mmHg)"),
    ]
    mapping_desc = [
        ("first_sensation", "初始感觉阈值 (ml)"),
        ("desire_to_defecate", "初始便意阈值 (ml)"),
        ("urgency_threshold", "排便窘迫感阈值 (ml)"),
        ("max_tolerable_volume", "最大容量感觉阈值 (ml)"),
        ("rair_min_volume", "RAIR 诱发最小容积 (ml)"),
        ("anal_length", "肛门括约肌长度 (cm)"),
    ]

    for key, label in mapping_core:
        if clinical.get(key) is not None:
            core_metrics[label] = clinical.get(key)

    for key, label in mapping_desc:
        if clinical.get(key) is not None:
            descriptive_metrics[label] = clinical.get(key)

    stability_label = "边界患者" if is_boundary else "稳定患者"

    patient_rag_features = build_patient_rag_features(clinical, gender=gender)
    rome_iv = build_rome_iv_proxy(clinical, gender=gender)

    protocol_contribution = get_patient_protocol_contribution(patient_id) or {
        "available": False,
        "message": "暂无协议级贡献结果。",
    }
    protocol_topk_details = get_patient_protocol_topk_details(patient_id) or []

    rair_features = get_patient_rair_features(patient_id) or {
        "available": False,
        "message": "暂无 RAIR 患者级特征。",
    }
    rair_ts_result = get_patient_rair_time_series(patient_id)
    rair_time_series = rair_ts_result.get("series") if isinstance(rair_ts_result, dict) else rair_ts_result
    rair_debug = rair_ts_result.get("debug", {}) if isinstance(rair_ts_result, dict) else {}

    rag_result = get_patient_rag_explanation(patient_rag_features, top_k=5)
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

    llm_key_findings = [
        f"分型置信度为 {confidence:.2f}",
        f"switch_rate 为 {switch_rate:.2f}",
        f"当前患者判定为：{stability_label}",
        f"Rome IV 代理分类：{rome_iv['category']}" if rome_iv["category"] else "Rome IV 代理分类：暂无",
        f"性别分层参考值：{gender}",
    ]

    if gender_meta.get("is_defaulted"):
        llm_key_findings.append("未检索到患者性别，当前使用默认 female 参考值，解释需谨慎。")

    if rag_explanation.get("summary"):
        llm_key_findings.append(f"知识库解释摘要：{rag_explanation['summary']}")

    clinical_significance = "当前页面已接入真实共识分型结果、临床联合表、协议级贡献信息与 RAIR 特征。"
    if rome_iv.get("proxy_type") == "simplified_proxy":
        clinical_significance += " Rome IV 部分为基于统一规则的简化代理分类，仅用于辅助解释。"
    if rag_explanation.get("interpretation"):
        clinical_significance += f" 基于文献知识库的解释提示：{rag_explanation['interpretation']}"

    recommendations = [
        "完整组间统计结果请查看 Statistics View。",
    ]
    if gender_meta.get("is_defaulted"):
        recommendations.append("建议优先补齐患者性别字段，以避免参考范围使用默认值带来的偏差。")
    if rag_explanation.get("uncertainty"):
        recommendations.append(f"解释不确定性：{rag_explanation['uncertainty']}")

    return {
        "patient_id": patient_id,
        "gender": gender,
        "gender_meta": gender_meta,
        "clinical_raw": clinical,
        "ai_result": {
            "cluster": cluster,
            "confidence": confidence,
            "is_boundary": is_boundary,
        },
        "physiology": {
            "core_metrics": core_metrics,
            "descriptive_metrics": descriptive_metrics,
        },
        "representation": {
            "protocol_contribution": protocol_contribution,
            "protocol_topk_details": protocol_topk_details,
        },
        "rair": {
            "time_series": rair_time_series,
            "features": rair_features,
            "debug": rair_debug,
            "meta": {
                "value_semantics": (
                    rair_features.get("value_semantics")
                    if isinstance(rair_features, dict)
                    else None
                ),
                "message": (
                    rair_features.get("message")
                    if isinstance(rair_features, dict)
                    else None
                ),
            },
        },
        "stability": {
            "confidence": confidence,
            "switch_rate": switch_rate,
            "is_boundary": is_boundary,
            "label": stability_label,
            "seed_assignments": get_patient_seed_assignments(patient_id) or {},
        },
        "rome_iv": rome_iv,
        "llm_analysis": {
            "summary": (
                rag_explanation.get("summary")
                or (f"该患者被分配到 Cluster {cluster}。" if cluster is not None else "该患者暂无可用分型结果。")
            ),
            "key_findings": llm_key_findings,
            "clinical_significance": clinical_significance,
            "recommendations": recommendations,
        },
        "group_statistics": {
            "version": "attn_pooling-8 / stable",
            "summary": "当前群体统计结果显示，稳定患者中共有 10 个指标进入 Kruskal-Wallis 分析，其中 6 个指标在 cluster 间具有显著差异。",
            "suggestion": "完整组间统计请查看 Statistics View。",
        },
        "rag": {
            "input_features": patient_rag_features,
            "retrieved_chunks": rag_chunks,
            "explanation": rag_explanation,
        },
        "rag_recommendations": rag_recommendations,
    }