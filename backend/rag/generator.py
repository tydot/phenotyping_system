from typing import Dict, List


def _unique_doc_ids(retrieved_chunks: List[Dict], limit: int = 3) -> List[str]:
    doc_ids = []
    seen = set()

    for c in retrieved_chunks:
        doc_id = str(c.get("doc_id", "")).strip()
        if not doc_id:
            continue
        if doc_id in seen:
            continue
        seen.add(doc_id)
        doc_ids.append(doc_id)
        if len(doc_ids) >= limit:
            break

    return doc_ids


def _append_if(parts: List[str], text: str):
    if text:
        parts.append(text)


def _build_patient_summary(features: Dict[str, str]) -> str:
    summary_parts: List[str] = []

    rp = features.get("resting_pressure")
    if rp == "low":
        _append_if(summary_parts, "患者表现为肛门括约肌静息压偏低")
    elif rp == "high":
        _append_if(summary_parts, "患者表现为肛门括约肌静息压偏高")
    elif rp == "normal":
        _append_if(summary_parts, "患者静息压大致处于正常范围")

    msp = features.get("msp")
    if msp == "low":
        _append_if(summary_parts, "伴随最大缩榨压偏低")
    elif msp == "high":
        _append_if(summary_parts, "伴随最大缩榨压偏高")
    elif msp == "normal":
        _append_if(summary_parts, "最大缩榨压大致正常")

    sqd = features.get("squeeze_duration")
    if sqd == "low":
        _append_if(summary_parts, "缩肛持续时间偏短")
    elif sqd == "high":
        _append_if(summary_parts, "缩肛持续时间偏长")
    elif sqd == "normal":
        _append_if(summary_parts, "缩肛持续时间大致正常")

    drp = features.get("defecatory_rectal_pressure")
    if drp == "low":
        _append_if(summary_parts, "提示排便推进力不足")
    elif drp == "high":
        _append_if(summary_parts, "排便时直肠压力偏高")
    elif drp == "normal":
        _append_if(summary_parts, "排便时直肠压力大致正常")

    fs = features.get("first_sensation")
    if fs == "high":
        _append_if(summary_parts, "初始感觉阈值升高")
    elif fs == "low":
        _append_if(summary_parts, "初始感觉阈值降低")
    elif fs == "normal":
        _append_if(summary_parts, "初始感觉阈值大致正常")

    ut = features.get("urge_threshold")
    if ut == "high":
        _append_if(summary_parts, "初始便意阈值升高")
    elif ut == "low":
        _append_if(summary_parts, "初始便意阈值降低")
    elif ut == "normal":
        _append_if(summary_parts, "初始便意阈值大致正常")

    dt = features.get("distress_threshold")
    if dt == "high":
        _append_if(summary_parts, "排便窘迫感阈值升高")
    elif dt == "low":
        _append_if(summary_parts, "排便窘迫感阈值降低")
    elif dt == "normal":
        _append_if(summary_parts, "排便窘迫感阈值大致正常")

    mtv = features.get("max_tolerable_volume")
    if mtv == "high":
        _append_if(summary_parts, "最大容量感觉阈值升高")
    elif mtv == "low":
        _append_if(summary_parts, "最大容量感觉阈值降低")
    elif mtv == "normal":
        _append_if(summary_parts, "最大容量感觉阈值大致正常")

    rs = features.get("rectal_sensation")
    if rs == "high_threshold":
        _append_if(summary_parts, "整体提示直肠感觉阈值偏高")
    elif rs == "low_threshold":
        _append_if(summary_parts, "整体提示直肠感觉阈值偏低")
    elif rs == "normal":
        _append_if(summary_parts, "整体直肠感觉大致正常")

    anal_length = features.get("anal_length")
    if anal_length == "low":
        _append_if(summary_parts, "肛门括约肌长度偏短")
    elif anal_length == "high":
        _append_if(summary_parts, "肛门括约肌长度偏长")
    elif anal_length == "normal":
        _append_if(summary_parts, "肛门括约肌长度大致正常")

    rair = features.get("rair")
    if rair == "normal":
        _append_if(summary_parts, "RAIR 大致正常")
    elif rair == "abnormal":
        _append_if(summary_parts, "RAIR 异常")

    if not summary_parts:
        return "当前未提供足够的结构化特征。"

    return "；".join(summary_parts) + "。"


def _build_patient_interpretation(features: Dict[str, str], retrieved_chunks: List[Dict]) -> str:
    hints = []

    if features.get("resting_pressure") == "low" and features.get("msp") == "low":
        hints.append("括约肌基础张力和主动收缩功能不足相关特征")

    if features.get("squeeze_duration") == "low":
        hints.append("收缩耐力不足相关特征")

    if features.get("defecatory_rectal_pressure") == "low":
        hints.append("推进不足相关特征")

    if features.get("rectal_sensation") == "high_threshold":
        hints.append("直肠感觉减退相关特征")

    if features.get("rectal_sensation") == "low_threshold":
        hints.append("直肠感觉敏化相关特征")

    if features.get("first_sensation") == "high" or features.get("urge_threshold") == "high":
        hints.append("感觉阈值升高相关表型")

    if features.get("distress_threshold") == "high" or features.get("max_tolerable_volume") == "high":
        hints.append("容量耐受增高相关表型")

    if features.get("rair") == "abnormal":
        hints.append("RAIR 异常相关表型")

    if features.get("anal_length") == "low":
        hints.append("肛管解剖长度偏短相关特征")
    elif features.get("anal_length") == "high":
        hints.append("肛管解剖长度偏长相关特征")

    if not hints:
        hints.append("已有文献中的相关功能异常表型")

    hint_text = "、".join(hints)

    if retrieved_chunks:
        top_doc_ids = _unique_doc_ids(retrieved_chunks, limit=3)
        doc_hint = f"主要参考文献包括：{', '.join(top_doc_ids)}。" if top_doc_ids else ""

        return (
            f"结合召回到的文献证据，该患者特征更接近{hint_text}。"
            f"{doc_hint}"
            "当前输出仅表示基于知识库证据的解释性说明，不构成临床诊断结论。"
        )

    return (
        "当前没有召回到足够的证据片段，暂时无法给出可靠解释。"
        "建议优先检查总表字段映射、tags、evidence_level 与 retrieval_priority 是否正确。"
    )


def generate_patient_explanation(patient_input: Dict, retrieved_chunks: List[Dict]) -> Dict:
    features = patient_input.get("features", {})
    summary = _build_patient_summary(features)
    interpretation = _build_patient_interpretation(features, retrieved_chunks)

    evidence = []
    for chunk in retrieved_chunks[:3]:
        evidence.append({
            "chunk_id": chunk.get("chunk_id", ""),
            "doc_id": chunk.get("doc_id", ""),
            "title": chunk.get("title", ""),
            "source": chunk.get("source", ""),
            "score": chunk.get("score", 0),
            "text": chunk.get("chunk_text", ""),
            "evidence_level": chunk.get("evidence_level", ""),
            "retrieval_priority": chunk.get("retrieval_priority", ""),
            "matched_terms": chunk.get("matched_terms", []),
            "matched_tags": chunk.get("matched_tags", []),
            "linked_core_doc": chunk.get("linked_core_doc", ""),
        })

    uncertainty = (
        "若缺少球囊排出试验、动态排便行为指标或完整感觉评估，"
        "则部分 Rome IV 相关判断仍存在不确定性。"
    )

    return {
        "summary": summary,
        "interpretation": interpretation,
        "uncertainty": uncertainty,
        "evidence": evidence,
    }


def _build_cluster_summary(cluster_input: Dict) -> str:
    cluster_id = cluster_input.get("cluster_id")
    cluster_name = cluster_input.get("cluster_name")
    summary_features = cluster_input.get("summary_features", [])

    display_name = cluster_name or (f"Cluster {cluster_id}" if cluster_id is not None else "当前 cluster")

    if summary_features:
        return f"{display_name} 的主要表型特征包括：" + "；".join(summary_features) + "。"

    return f"{display_name} 当前未提供明确的 summary_features。"


def _build_cluster_interpretation(cluster_input: Dict, retrieved_chunks: List[Dict]) -> str:
    cluster_id = cluster_input.get("cluster_id")
    cluster_name = cluster_input.get("cluster_name")
    summary_features = [str(x).strip() for x in cluster_input.get("summary_features", []) if str(x).strip()]

    display_name = cluster_name or (f"Cluster {cluster_id}" if cluster_id is not None else "当前 cluster")

    hints = []

    text_blob = " ".join(summary_features).lower()

    if any(x in text_blob for x in ["poor propulsion", "low rectal propulsive force"]):
        hints.append("推进力不足")
    if any(x in text_blob for x in ["weak squeeze", "low squeeze pressure", "sphincter weakness"]):
        hints.append("主动收缩能力减弱")
    if any(x in text_blob for x in ["hyposensitivity", "sensory dysfunction", "high sensory threshold"]):
        hints.append("直肠感觉减退")
    if any(x in text_blob for x in ["dyssynergia", "defecatory dysfunction"]):
        hints.append("排便协调障碍相关表型")
    if any(x in text_blob for x in ["biofeedback candidate", "biofeedback relevance"]):
        hints.append("可能与生物反馈治疗适应性相关")

    if not hints and summary_features:
        hints.append("相对一致的群体表型特征")

    hint_text = "、".join(hints) if hints else "相对一致的群体表型特征"

    if retrieved_chunks:
        top_doc_ids = _unique_doc_ids(retrieved_chunks, limit=3)
        doc_hint = f"主要参考文献包括：{', '.join(top_doc_ids)}。" if top_doc_ids else ""

        return (
            f"结合召回证据，{display_name} 可能代表一个以{hint_text}为主的亚型。"
            "其解释应结合压力、感觉、RAIR、排便推进及治疗反应等维度综合理解。"
            f"{doc_hint}"
            "该结果属于 cluster-level 的知识支持性解释，不应直接替代单个患者的临床判断。"
        )

    return (
        f"{display_name} 当前没有召回到足够证据，暂时不能给出稳定解释。"
        "建议优先检查 cluster summary_features、tags、subkb_type 和 priority 设置。"
    )


def generate_cluster_explanation(cluster_input: Dict, retrieved_chunks: List[Dict]) -> Dict:
    summary = _build_cluster_summary(cluster_input)
    interpretation = _build_cluster_interpretation(cluster_input, retrieved_chunks)

    evidence = []
    for chunk in retrieved_chunks[:3]:
        evidence.append({
            "chunk_id": chunk.get("chunk_id", ""),
            "doc_id": chunk.get("doc_id", ""),
            "title": chunk.get("title", ""),
            "source": chunk.get("source", ""),
            "score": chunk.get("score", 0),
            "text": chunk.get("chunk_text", ""),
            "evidence_level": chunk.get("evidence_level", ""),
            "retrieval_priority": chunk.get("retrieval_priority", ""),
            "matched_terms": chunk.get("matched_terms", []),
            "matched_tags": chunk.get("matched_tags", []),
            "linked_core_doc": chunk.get("linked_core_doc", ""),
        })

    return {
        "summary": summary,
        "interpretation": interpretation,
        "uncertainty": "cluster 解释属于群体层面的归纳，不应直接替代个体患者诊断。",
        "evidence": evidence,
    }