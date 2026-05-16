# backend/vlm/consistency_gate.py

from typing import Dict, Any, List


REGION_TO_FEATURE_KEYWORDS = {
    "resting_phase": [
        "resting",
        "rest",
        "静息",
        "静息压",
        "肛门静息压",
        "resting_pressure",
        "RestPressure",
        "ARP",
    ],
    "squeeze_phase": [
        "squeeze",
        "squeeze_pressure",
        "缩榨",
        "缩肛",
        "最大缩榨压",
        "主动收缩",
        "MSP",
    ],
    "defecation_phase": [
        "defecation",
        "defecate",
        "排便",
        "排便模拟",
        "直肠推进",
        "推进力",
        "rectal_pressure",
        "push",
        "propulsion",
    ],
    "rair_phase": [
        "rair",
        "RAIR",
        "松弛反射",
        "肛门直肠抑制反射",
        "relaxation",
        "recovery",
    ],
}


ABNORMAL_WORDS = [
    "high",
    "low",
    "abnormal",
    "decreased",
    "increased",
    "reduced",
    "elevated",
    "降低",
    "升高",
    "偏低",
    "偏高",
    "不足",
    "异常",
    "减弱",
    "增强",
    "缩短",
    "延长",
    "缺失",
]


NORMAL_WORDS = [
    "normal",
    "正常",
    "无异常",
]


def _to_text(value: Any) -> str:
    """
    将任意结构转成便于关键词匹配的文本。
    """
    if value is None:
        return ""

    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            parts.append(str(k))
            parts.append(_to_text(v))
        return " ".join(parts)

    if isinstance(value, list):
        return " ".join(_to_text(x) for x in value)

    return str(value)


def _normalize_feature_states(feature_states: Any) -> Dict[str, Any]:
    """
    将 feature_states 统一成 dict，避免外部传入 list 时 .items() 报错。
    """
    if isinstance(feature_states, dict):
        return feature_states

    result: Dict[str, Any] = {}

    if isinstance(feature_states, list):
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

    elif feature_states is not None:
        result["feature_states"] = feature_states

    return result


def _has_keyword(text: str, keywords: List[str]) -> bool:
    text_lower = text.lower()
    for kw in keywords:
        if str(kw).lower() in text_lower:
            return True
    return False


def _has_abnormal_word(text: str) -> bool:
    return _has_keyword(text, ABNORMAL_WORDS)


def _has_normal_word(text: str) -> bool:
    return _has_keyword(text, NORMAL_WORDS)


def _has_related_abnormality(region_id: str, feature_states: Any) -> bool:
    """
    判断结构化指标里是否存在与当前图像区域相关的异常。
    """
    states = _normalize_feature_states(feature_states)
    keywords = REGION_TO_FEATURE_KEYWORDS.get(region_id, [])

    for key, value in states.items():
        combined_text = f"{key} {_to_text(value)}"

        if not _has_keyword(combined_text, keywords):
            continue

        if _has_abnormal_word(combined_text):
            return True

    return False


def _has_related_normal(region_id: str, feature_states: Any) -> bool:
    """
    判断结构化指标里是否存在与当前图像区域相关的正常信息。
    """
    states = _normalize_feature_states(feature_states)
    keywords = REGION_TO_FEATURE_KEYWORDS.get(region_id, [])

    for key, value in states.items():
        combined_text = f"{key} {_to_text(value)}"

        if not _has_keyword(combined_text, keywords):
            continue

        if _has_normal_word(combined_text):
            return True

    return False


def check_one_finding(
    finding: Dict[str, Any],
    feature_states: Any,
) -> Dict[str, Any]:
    """
    对单个图像区域发现进行一致性门控。

    输出字段：
    - consistency_status:
        consistent
        conflict
        uncertain
        weak_visual_evidence
        consistent_negative
    - use_in_report:
        True / False
    - consistency_note:
        给 patient.py 和 LLM 使用的解释说明
    """
    finding = dict(finding or {})

    region_id = finding.get("region_id")
    visual_support = str(finding.get("visual_support", "")).strip().lower()

    related_abnormal = _has_related_abnormality(region_id, feature_states)
    related_normal = _has_related_normal(region_id, feature_states)

    if visual_support == "support" and related_abnormal:
        status = "consistent"
        use_in_report = True
        note = "图像侧发现与结构化功能异常方向一致，可作为辅助证据。"

    elif visual_support == "support" and not related_abnormal:
        status = "conflict"
        use_in_report = False
        note = "图像侧提示异常，但结构化指标未见对应异常，不作为主要结论依据。"

    elif visual_support == "uncertain":
        status = "uncertain"
        use_in_report = False
        note = "图像侧证据不足，仅展示，不作为结论依据。"

    elif visual_support == "not_support" and related_abnormal:
        status = "weak_visual_evidence"
        use_in_report = False
        note = "结构化指标提示异常，但图像侧未见明确支持证据。"

    elif visual_support == "not_support" and related_normal:
        status = "consistent_negative"
        use_in_report = True
        note = "结构化指标和图像侧均未提示明确异常，可作为阴性辅助说明。"

    elif visual_support == "not_support":
        status = "consistent_negative"
        use_in_report = False
        note = "图像侧未见明确异常，但缺少足够结构化指标对应关系，暂不进入报告。"

    else:
        status = "uncertain"
        use_in_report = False
        note = "图像侧输出格式不明确，暂不作为结论依据。"

    finding["consistency_status"] = status
    finding["use_in_report"] = use_in_report
    finding["consistency_note"] = note
    finding["related_structured_abnormality"] = related_abnormal
    finding["related_structured_normal"] = related_normal

    return finding


def check_visual_clinical_consistency(
    region_findings: List[Dict[str, Any]],
    feature_states: Any,
) -> List[Dict[str, Any]]:
    """
    批量检查图像侧区域解释与结构化功能指标的一致性。
    """
    if not region_findings:
        return []

    checked_findings = []

    for finding in region_findings:
        checked = check_one_finding(
            finding=finding,
            feature_states=feature_states,
        )
        checked_findings.append(checked)

    return checked_findings


__all__ = [
    "check_one_finding",
    "check_visual_clinical_consistency",
]