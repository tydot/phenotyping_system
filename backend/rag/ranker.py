from typing import Dict, Any, Optional


MALE_REF = {
    "resting_pressure": (60, 120),
    "msp": (150, 300),
    "sensation_first": (10, 30),
    "sensation_urge": (40, 90),
    "sensation_distress": (80, 180),
    "sensation_max": (150, 300),
    "rair_volume": (10, 30),
    "rectal_pressure": (45, 120),
    "squeeze_duration": (10, 25),
    "anal_length": (3.0, 5.0),
}

FEMALE_REF = {
    "resting_pressure": (50, 100),
    "msp": (120, 250),
    "sensation_first": (10, 30),
    "sensation_urge": (30, 80),
    "sensation_distress": (70, 160),
    "sensation_max": (120, 280),
    "rair_volume": (10, 30),
    "rectal_pressure": (45, 120),
    "squeeze_duration": (10, 25),
    "anal_length": (2.5, 4.5),
}


def _safe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def normalize_gender(gender: Any) -> str:
    g = str(gender or "").strip().lower()
    if g in {"male", "m", "man", "男"}:
        return "male"
    if g in {"female", "f", "woman", "女"}:
        return "female"
    return "female"


def get_reference_by_gender(gender: Any) -> Dict[str, tuple]:
    return MALE_REF if normalize_gender(gender) == "male" else FEMALE_REF


def judge_abnormal(value: Optional[float], ref_range: tuple) -> Optional[str]:
    if value is None:
        return None
    low, high = ref_range
    if value < low:
        return "low"
    if value > high:
        return "high"
    return "normal"


def build_patient_feature_states(clinical: Dict[str, Any], gender: Any) -> Dict[str, str]:
    """
    用统一临床规则生成患者特征状态。
    返回结果用于：
    1) RAG 检索输入
    2) KG 特征映射
    """
    if not clinical:
        return {}

    ref = get_reference_by_gender(gender)

    rp = _safe_float(clinical.get("resting_pressure"))
    msp = _safe_float(clinical.get("msp"))
    squeeze_duration = _safe_float(clinical.get("squeeze_duration"))
    rectal_pressure = _safe_float(clinical.get("defecatory_rectal_pressure"))
    first_sensation = _safe_float(clinical.get("first_sensation"))
    urge_threshold = _safe_float(clinical.get("desire_to_defecate"))
    distress_threshold = _safe_float(clinical.get("urgency_threshold"))
    max_tolerable_volume = _safe_float(clinical.get("max_tolerable_volume"))
    rair_min_volume = _safe_float(clinical.get("rair_min_volume"))
    anal_length = _safe_float(clinical.get("anal_length"))

    features: Dict[str, str] = {}

    s = judge_abnormal(rp, ref["resting_pressure"])
    if s:
        features["resting_pressure"] = s

    s = judge_abnormal(msp, ref["msp"])
    if s:
        features["msp"] = s

    s = judge_abnormal(squeeze_duration, ref["squeeze_duration"])
    if s:
        features["squeeze_duration"] = s

    s = judge_abnormal(rectal_pressure, ref["rectal_pressure"])
    if s:
        features["defecatory_rectal_pressure"] = s

    s = judge_abnormal(first_sensation, ref["sensation_first"])
    if s:
        features["first_sensation"] = s

    s = judge_abnormal(urge_threshold, ref["sensation_urge"])
    if s:
        features["urge_threshold"] = s

    s = judge_abnormal(distress_threshold, ref["sensation_distress"])
    if s:
        features["distress_threshold"] = s

    s = judge_abnormal(max_tolerable_volume, ref["sensation_max"])
    if s:
        features["max_tolerable_volume"] = s

    s = judge_abnormal(anal_length, ref["anal_length"])
    if s:
        features["anal_length"] = s

    # RAIR 特殊处理
    if rair_min_volume is not None:
        rair_status = judge_abnormal(rair_min_volume, ref["rair_volume"])
        if rair_status == "high":
            features["rair"] = "abnormal"
        else:
            features["rair"] = "normal"

    # 聚合感觉功能，方便 query_builder 兼容旧检索语义
    sensation_flags = []
    for key in ["first_sensation", "urge_threshold", "distress_threshold", "max_tolerable_volume"]:
        value = features.get(key)
        if value:
            sensation_flags.append(value)

    if any(v == "high" for v in sensation_flags):
        features["rectal_sensation"] = "high_threshold"
    elif any(v == "low" for v in sensation_flags):
        features["rectal_sensation"] = "low_threshold"
    elif sensation_flags:
        features["rectal_sensation"] = "normal"

    return features