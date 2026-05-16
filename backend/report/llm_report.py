# backend/report/llm_report.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List
import math


PHENOTYPE_MAP = {
    "0": {
        "phenotype_name": "轻-中度短肛管伴感觉阈值降低型",
        "short_name": "轻-中度短肛管-感觉敏感型",
        "description": (
            "主要表现为肛门括约肌长度偏短，并伴随初始便意阈值、"
            "排便窘迫感阈值和最大容量感觉阈值降低。静息压降低程度相对较轻。"
        ),
    },
    "1": {
        "phenotype_name": "重度低静息张力-短肛管伴收缩/推进不足型",
        "short_name": "重度括约肌低功能型",
        "description": (
            "三类亚群中功能低下最明显，常表现为静息压、肛门括约肌长度、"
            "最大缩榨压、缩肛持续时间和排便时直肠压力降低。"
        ),
    },
    "2": {
        "phenotype_name": "低容量耐受-感觉敏感伴短肛管型",
        "short_name": "容量耐受下降-感觉敏感型",
        "description": (
            "核心特征是容量耐受下降和感觉敏感，表现为最大容量感觉阈值、"
            "初始便意阈值和排便窘迫感阈值降低，同时伴随短肛管。"
        ),
    },
}


def safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def normalize_cluster(cluster: Any) -> str:
    if cluster is None:
        return "unknown"

    try:
        return str(int(float(cluster)))
    except Exception:
        return str(cluster)


def fmt_value(value: Any) -> str:
    if value is None or value == "":
        return "-"

    try:
        x = float(value)
        if math.isinf(x):
            return "∞"
        return f"{x:.2f}"
    except Exception:
        return str(value)


def get_phenotype_info(cluster: Any) -> Dict[str, str]:
    key = normalize_cluster(cluster)
    return PHENOTYPE_MAP.get(
        key,
        {
            "phenotype_name": f"Cluster {cluster}",
            "short_name": f"Cluster {cluster}",
            "description": "当前簇尚未配置正式表型命名。",
        },
    )


def status_to_cn(status: Any) -> str:
    s = str(status).lower().strip()

    mapping = {
        "low": "低于参考范围",
        "high": "高于参考范围",
        "normal": "处于参考范围",
        "missing": "指标缺失",
        "no_reference": "暂无参考范围",
    }

    return mapping.get(s, str(status))


def extract_abnormal_metrics(metric_judgements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    只把异常、缺失、无参考范围的指标交给解释报告重点展示。
    normal 指标不作为关键发现。
    """
    abnormal = []

    for item in safe_list(metric_judgements):
        item = safe_dict(item)
        status = str(item.get("status", "")).lower().strip()

        if status in {"low", "high", "missing", "no_reference"}:
            abnormal.append(item)

    return abnormal


def build_llm_context(
    patient_id: str,
    ai: Dict[str, Any],
    stability: Dict[str, Any],
    metric_judgements: List[Dict[str, Any]],
    feature_states: List[str] | None,
    rair: Dict[str, Any],
    rome: Dict[str, Any],
    rag: Dict[str, Any],
    kg_paths: List[Any] | None = None,
) -> Dict[str, Any]:
    """
    构建 LLM/解释模块输入。

    边界：
    - 不放患者姓名；
    - 不放本地原始图像路径；
    - 不放 API Key；
    - 不让 LLM 重新判断指标正常/异常；
    - 指标状态完全来自 feature_state_extractor.py。
    """
    ai = safe_dict(ai)
    stability = safe_dict(stability)
    rair = safe_dict(rair)
    rome = safe_dict(rome)
    rag = safe_dict(rag)

    cluster = ai.get("cluster")
    phenotype = get_phenotype_info(cluster)

    return {
        "patient_id": str(patient_id),
        "ai_result": {
            "cluster": cluster,
            "confidence": ai.get("confidence"),
            "is_boundary": ai.get("is_boundary"),
            "stability_label": stability.get("label"),
            "switch_rate": stability.get("switch_rate"),
        },
        "phenotype": phenotype,
        "metric_judgements": safe_list(metric_judgements),
        "abnormal_metrics": extract_abnormal_metrics(metric_judgements),
        "feature_states": safe_list(feature_states),
        "rair_features": safe_dict(rair.get("features")),
        "rome_iv": rome,
        "rag": {
            "input_features": safe_dict(rag.get("input_features")),
            "retrieved_chunks": safe_list(rag.get("retrieved_chunks")),
            "explanation": safe_dict(rag.get("explanation")),
        },
        "kg_paths": safe_list(kg_paths),
    }


def build_reference_text(item: Dict[str, Any]) -> str:
    low = item.get("low")
    high = item.get("high")
    center = item.get("center")
    reference_group = item.get("reference_group") or ""
    sex = item.get("sex") or "-"

    parts = []

    if low is not None or high is not None:
        try:
            high_float = float(high)
            if math.isinf(high_float):
                parts.append(f"参考范围为 ≥{fmt_value(low)}")
            else:
                parts.append(f"参考范围约为 {fmt_value(low)}–{fmt_value(high)}")
        except Exception:
            parts.append(f"参考范围约为 {fmt_value(low)}–{fmt_value(high)}")

    if center is not None:
        parts.append(f"参考中心为 {fmt_value(center)}")

    if reference_group:
        parts.append(f"依据：{reference_group}")
    elif sex in {"M", "F"}:
        parts.append(f"性别匹配：{sex}")

    if not parts:
        return ""

    return "，" + "，".join(parts)


def generate_rule_based_report(context: Dict[str, Any]) -> str:
    """
    规则版科研解释报告。

    当前用于先把 patient.py 页面跑通。
    后续接真实 LLM API 时，可以保持 context 不变，只替换该函数。
    """
    context = safe_dict(context)

    ai = safe_dict(context.get("ai_result"))
    phenotype = safe_dict(context.get("phenotype"))
    abnormal_metrics = safe_list(context.get("abnormal_metrics"))
    feature_states = safe_list(context.get("feature_states"))
    rag = safe_dict(context.get("rag"))
    rag_chunks = safe_list(rag.get("retrieved_chunks"))
    kg_paths = safe_list(context.get("kg_paths"))
    rair_features = safe_dict(context.get("rair_features"))
    rome = safe_dict(context.get("rome_iv"))

    cluster = ai.get("cluster")
    confidence = ai.get("confidence")
    is_boundary = bool(ai.get("is_boundary"))
    phenotype_name = phenotype.get("phenotype_name", f"Cluster {cluster}")
    phenotype_desc = phenotype.get("description", "")

    if is_boundary:
        stability_text = "该患者被标记为边界患者，提示其分型在不同随机种子下可能存在一定波动。"
    else:
        stability_text = "该患者属于稳定分配患者，提示其在多随机种子聚类中具有较一致的簇归属。"

    if abnormal_metrics:
        finding_lines = []

        for item in abnormal_metrics[:8]:
            item = safe_dict(item)

            metric = item.get("metric") or "-"
            value = item.get("value")
            status = status_to_cn(item.get("status"))
            explanation = item.get("state_text") or ""
            ref_text = build_reference_text(item)

            finding_lines.append(
                f"- **{metric}**：{status}，患者数值为 {fmt_value(value)}{ref_text}。{explanation}"
            )

        key_findings = "\n".join(finding_lines)
    else:
        key_findings = "- 当前结构化输入中未发现明确异常指标，建议结合完整测压报告继续核查。"

    if feature_states:
        feature_state_text = "；".join([str(x) for x in feature_states[:8]])
    else:
        feature_state_text = "当前未生成明确功能状态标签。"

    if rair_features and rair_features.get("available") is not False:
        relaxation = rair_features.get("relaxation_amplitude")
        t_min = rair_features.get("t_min")

        rair_text = (
            f"当前患者存在 RAIR 相关特征记录，其中松弛幅度为 {fmt_value(relaxation)}，"
            f"达到最低点时间为 {fmt_value(t_min)}。"
        )
    else:
        rair_text = "当前患者暂无可用于解释的 RAIR 患者级特征。"

    if rome and rome.get("category") is not None:
        rome_text = (
            f"Rome IV 代理分类结果为 {rome.get('category')}，"
            f"推进力状态为 {rome.get('propulsion', '-') or '-'}，"
            f"协调性状态为 {rome.get('coordination', '-') or '-'}。"
        )
    else:
        rome_text = "当前暂无 Rome IV 代理分类信息。"

    if rag_chunks:
        evidence_lines = []

        for idx, chunk in enumerate(rag_chunks[:3], 1):
            chunk = safe_dict(chunk)

            title = chunk.get("title") or chunk.get("chunk_id") or f"证据 {idx}"
            source = chunk.get("source") or "未知来源"
            matched_terms = chunk.get("matched_terms") or []

            evidence_lines.append(
                f"- {title}（来源：{source}；匹配词：{matched_terms if matched_terms else '未提供'}）"
            )

        evidence_text = "\n".join(evidence_lines)
    else:
        evidence_text = "当前未召回充分文献证据，因此文献支持部分仍需后续补充。"

    if kg_paths:
        kg_text = "系统已生成知识图谱解释路径，可用于连接患者特征、功能表型、可能机制和证据节点。"
    else:
        kg_text = "当前未获得可用知识图谱解释路径，机制解释主要依据结构化指标和表型画像。"

    uncertainty_items = [
        "本报告仅用于科研辅助分析，不构成临床诊断或治疗建议。",
        "LLM 未参与患者分型，患者分型仍由无监督共识聚类模型给出。",
        "指标正常/异常状态由医院报告参考范围和性别匹配规则确定，LLM 不重新判断参考范围。",
        "若存在指标缺失、RAG 证据不足或知识图谱路径缺失，应结合人工复核。",
    ]

    if is_boundary:
        uncertainty_items.append("该患者属于边界样本，分型解释需要更谨慎。")

    uncertainty_text = "\n".join([f"- {x}" for x in uncertainty_items])

    return f"""
### 分析摘要
该患者被当前 AI 分型模型分配至 **{phenotype_name}**，对应 **Cluster {cluster}**，稳定性置信度为 **{fmt_value(confidence)}**。{stability_text}

{phenotype_desc}

### 关键发现
{key_findings}

### 可能机制
结合该患者的 AI 表型、医院参考范围判定和群体画像结果，当前患者可能存在与 **{phenotype_name}** 相关的功能改变。结构化功能状态包括：{feature_state_text}

{rair_text} {rome_text}

{kg_text}

### 文献支持
{evidence_text}

### 科研建议
- 建议将该患者的异常指标与同 Cluster 群体画像进行对照，观察其是否符合该亚型的主要表型特征。
- 建议结合 RAIR 特征、Rome IV proxy 和组间统计结果进行综合解释。
- 若后续引入医生复核或专家标注，可进一步验证该患者是否属于典型表型或过渡表型。

### 不确定性说明
{uncertainty_text}
""".strip()