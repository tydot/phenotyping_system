from typing import Dict, List


def deduplicate_keep_order(items: List[str]) -> List[str]:
    seen = set()
    result = []

    for item in items:
        item = str(item).strip()
        if not item:
            continue
        low = item.lower()
        if low not in seen:
            seen.add(low)
            result.append(item)

    return result


def _expand_term(term: str) -> List[str]:
    t = str(term).strip().lower()
    if not t:
        return []

    mapping = {
        "poor propulsion": [
            "poor propulsion",
            "propulsion",
            "defecatory disorder",
            "defecatory dysfunction",
            "low rectal propulsive force",
            "propulsive force",
        ],
        "low rectal propulsive force": [
            "low rectal propulsive force",
            "poor propulsion",
            "propulsion",
            "defecatory dysfunction",
        ],
        "dyssynergia": [
            "dyssynergia",
            "defecatory disorder",
            "functional defecation disorder",
            "coordination disorder",
            "classification",
        ],
        "weak squeeze": [
            "weak squeeze",
            "squeeze pressure",
            "low squeeze pressure",
            "sphincter weakness",
            "anal sphincter weakness",
        ],
        "sphincter weakness": [
            "sphincter weakness",
            "anal sphincter weakness",
            "low squeeze pressure",
            "low resting pressure",
        ],
        "low resting pressure": [
            "low resting pressure",
            "resting pressure",
            "low resting tone",
            "sphincter weakness",
        ],
        "high resting pressure": [
            "high resting pressure",
            "resting pressure",
            "hypertonic",
            "high anal tone",
        ],
        "hyposensitivity": [
            "hyposensitivity",
            "rectal hyposensitivity",
            "sensory dysfunction",
            "rectal sensory dysfunction",
            "high sensory threshold",
        ],
        "hypersensitivity": [
            "hypersensitivity",
            "rectal hypersensitivity",
            "sensory dysfunction",
            "low sensory threshold",
        ],
        "rair": [
            "RAIR",
            "rectoanal inhibitory reflex",
        ],
        "absent rair": [
            "RAIR",
            "rectoanal inhibitory reflex",
            "absent RAIR",
            "aganglionosis",
        ],
        "biofeedback candidate": [
            "biofeedback",
            "biofeedback candidate",
            "first-line treatment",
            "therapy",
        ],
        "short squeeze duration": [
            "short squeeze duration",
            "reduced squeeze endurance",
            "impaired voluntary contraction",
        ],
        "long anal canal": [
            "anal length",
            "long anal canal",
            "anorectal anatomy",
        ],
        "short anal canal": [
            "anal length",
            "short anal canal",
            "anorectal anatomy",
        ],
        "urge sensation impairment": [
            "urge threshold",
            "impaired urge sensation",
            "rectal sensory dysfunction",
            "hyposensitivity",
        ],
        "distress sensation impairment": [
            "urgency threshold",
            "distress threshold",
            "rectal sensory dysfunction",
            "hyposensitivity",
        ],
        "increased rectal capacity": [
            "max tolerable volume",
            "rectal capacity",
            "high sensory threshold",
            "hyposensitivity",
        ],
    }

    return mapping.get(t, [term])


def build_patient_query_terms(features: Dict[str, str]) -> List[str]:
    """
    将统一后的患者临床特征状态映射为知识库检索词。
    """

    terms: List[str] = []

    rp = features.get("resting_pressure")
    msp = features.get("msp")
    drp = features.get("defecatory_rectal_pressure")
    rs = features.get("rectal_sensation")
    rair = features.get("rair")
    squeeze_duration = features.get("squeeze_duration")
    first_sensation = features.get("first_sensation")
    urge_threshold = features.get("urge_threshold")
    distress_threshold = features.get("distress_threshold")
    max_tolerable_volume = features.get("max_tolerable_volume")
    anal_length = features.get("anal_length")

    if drp == "low":
        terms.extend([
            "propulsion",
            "poor propulsion",
            "defecatory disorder",
            "defecatory dysfunction",
            "dyssynergia",
            "propulsive force",
            "low rectal propulsive force",
        ])
    elif drp == "high":
        terms.extend([
            "propulsion",
            "defecatory rectal pressure",
            "high rectal propulsive force",
        ])
    elif drp == "normal":
        terms.extend([
            "propulsion",
            "defecatory rectal pressure",
        ])

    if rp == "low":
        terms.extend([
            "sphincter weakness",
            "anal sphincter weakness",
            "hypotonic",
            "resting pressure",
            "low resting pressure",
            "low resting tone",
        ])
    elif rp == "high":
        terms.extend([
            "resting pressure",
            "hypertonic",
            "high resting pressure",
            "high anal tone",
        ])
    elif rp == "normal":
        terms.extend(["resting pressure"])

    if msp == "low":
        terms.extend([
            "sphincter weakness",
            "squeeze pressure",
            "anal sphincter weakness",
            "low squeeze pressure",
            "weak squeeze",
            "impaired voluntary contraction",
        ])
    elif msp == "high":
        terms.extend([
            "squeeze pressure",
            "high squeeze pressure",
        ])
    elif msp == "normal":
        terms.extend(["squeeze pressure"])

    if squeeze_duration == "low":
        terms.extend([
            "short squeeze duration",
            "reduced squeeze endurance",
            "impaired voluntary contraction",
        ])

    # 细化感觉轴
    if first_sensation == "high":
        terms.extend([
            "rectal hyposensitivity",
            "high sensory threshold",
            "sensory dysfunction",
        ])
    elif first_sensation == "low":
        terms.extend([
            "rectal hypersensitivity",
            "low sensory threshold",
            "sensory dysfunction",
        ])

    if urge_threshold == "high":
        terms.extend([
            "urge sensation impairment",
            "impaired urge sensation",
            "hyposensitivity",
            "rectal sensory dysfunction",
        ])
    elif urge_threshold == "low":
        terms.extend([
            "urge threshold",
            "rectal hypersensitivity",
            "sensory dysfunction",
        ])

    if distress_threshold == "high":
        terms.extend([
            "distress sensation impairment",
            "urgency threshold",
            "hyposensitivity",
            "rectal sensory dysfunction",
        ])
    elif distress_threshold == "low":
        terms.extend([
            "urgency threshold",
            "hypersensitivity",
            "rectal sensory dysfunction",
        ])

    if max_tolerable_volume == "high":
        terms.extend([
            "increased rectal capacity",
            "max tolerable volume",
            "high sensory threshold",
            "hyposensitivity",
        ])
    elif max_tolerable_volume == "low":
        terms.extend([
            "decreased rectal capacity",
            "max tolerable volume",
            "hypersensitivity",
        ])

    # 聚合感觉项，兼容旧知识库标签
    if rs == "high_threshold":
        terms.extend([
            "rectal sensory dysfunction",
            "hyposensitivity",
            "sensory dysfunction",
            "rectal hyposensitivity",
            "high sensory threshold",
        ])
    elif rs == "low_threshold":
        terms.extend([
            "rectal sensory dysfunction",
            "hypersensitivity",
            "sensory dysfunction",
            "rectal hypersensitivity",
            "low sensory threshold",
        ])
    elif rs == "normal":
        terms.extend(["rectal sensation"])

    if rair == "normal":
        terms.extend([
            "RAIR",
            "rectoanal inhibitory reflex",
        ])
    elif rair == "abnormal":
        terms.extend([
            "RAIR",
            "rectoanal inhibitory reflex",
            "absent RAIR",
            "RAIR abnormality",
            "aganglionosis",
        ])

    if anal_length == "low":
        terms.extend([
            "anal length",
            "short anal canal",
            "anorectal anatomy",
        ])
    elif anal_length == "high":
        terms.extend([
            "anal length",
            "long anal canal",
            "anorectal anatomy",
        ])
    elif anal_length == "normal":
        terms.extend(["anal length"])

    terms.extend([
        "anorectal manometry",
        "functional defecation disorder",
        "anorectal disorder",
    ])

    return deduplicate_keep_order(terms)


def build_cluster_query_terms(summary_features: List[str]) -> List[str]:
    terms: List[str] = []

    for item in summary_features:
        item = str(item).strip()
        if not item:
            continue
        terms.append(item)
        terms.extend(_expand_term(item))

    terms.extend([
        "phenotype",
        "pathophysiology",
        "classification",
        "subtype",
        "anorectal manometry",
        "functional defecation disorder",
        "anorectal disorder",
    ])

    if not summary_features:
        terms.extend([
            "phenotype",
            "classification",
            "pathophysiology",
            "dyssynergia",
            "poor propulsion",
            "sphincter weakness",
            "sensory dysfunction",
            "RAIR",
        ])

    return deduplicate_keep_order(terms)