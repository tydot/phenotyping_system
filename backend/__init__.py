# backend/vlm/consistency_gate.py

from typing import Dict, Any, List


REGION_TO_FEATURE_KEYWORDS = {
    "resting_phase": ["resting", "静息压", "肛门静息压", "ARP"],
    "squeeze_phase": ["squeeze", "缩榨", "最大缩榨压", "MSP"],
    "defecation_phase": ["defecation", "排便", "直肠推进", "推进力", "rectal_pressure"],
    "rair_phase": ["rair", "RAIR", "松弛反射"],
}


def _has_related_abnormality(region_id: str, feature_states: Dict[str, Any]) -> bool:
    keywords = REGION_TO_FEATURE_KEYWORDS.get(region_id, [])

    for key, value in feature_states.items():
        text = f"{key} {value}".lower()
        for kw in keywords:
            if kw.lower() in text:
                if any(flag in text for flag in ["high", "low", "abnormal", "异常", "降低", "升高", "不足"]):
                    return True

    return False


def check_one_finding(
    finding: Dict[str, Any],
    feature_states: Dict[str, Any],
) -> Dict[str, Any]:
    region_id = finding.get("region_id")
    visual_support = finding.get("visual_support")

    related_abnormal = _has_related_abnormality(region_id, feature_states)

    if visual_support == "support" and related_abnormal:
        status = "consistent"
        use_in_report = True
        note = "图像侧发现与结构化功能异常方向一致，可作为辅助证据。"
    elif visual_support == "support" and not related_abnormal:
        status = "conflict"
        use_in_report = False
        note = "图像侧提示异常，但结构化指标未见对应异常，不作为结论依据。"
    elif visual_support == "uncertain":
        status = "uncertain"
        use_in_report = False
        note = "图像侧证据不足，仅展示，不作为结论依据。"
    elif visual_support == "not_support" and related_abnormal:
        status = "weak_visual_evidence"
        use_in_report = False
        note = "结构化指标提示异常，但图像侧未见明确支持证据。"
    else:
        status = "consistent_negative"
        use_in_report = True
        note = "结构化指标和图像侧均未提示明确异常。"

    checked = dict(finding)
    checked["consistency_status"] = status
    checked["use_in_report"] = use_in_report
    checked["consistency_note"] = note
    return checked


def check_visual_clinical_consistency(
    region_findings: List[Dict[str, Any]],
    feature_states: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return [
        check_one_finding(finding, feature_states)
        for finding in region_findings
    ]