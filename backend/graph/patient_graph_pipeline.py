from typing import Dict, Any, List

from backend.api.patient import get_patient_view
from backend.clinical.feature_rules import build_patient_feature_states, normalize_gender


def _safe_dict(value):
    return value if isinstance(value, dict) else {}


def _dedup_nodes(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    seen = set()
    for node in nodes:
        node_id = str(node.get("id", "")).strip()
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        out.append(node)
    return out


def _dedup_edges(edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    seen = set()
    for edge in edges:
        key = (
            str(edge.get("source", "")).strip(),
            str(edge.get("target", "")).strip(),
            str(edge.get("relation", "")).strip(),
        )
        if not all(key) or key in seen:
            continue
        seen.add(key)
        out.append(edge)
    return out


def _make_node(node_id: str, label: str, node_type: str, properties: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "id": node_id,
        "label": label,
        "type": node_type,
        "properties": properties or {},
    }


def _make_edge(source: str, target: str, relation: str, properties: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "source": source,
        "target": target,
        "relation": relation,
        "properties": properties or {},
    }


def build_observed_feature_nodes(patient: Dict[str, Any]):
    """
    统一复用 backend.clinical.feature_rules.build_patient_feature_states()
    保证与 RAG 的特征判定完全一致。
    """
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    context: List[str] = []

    patient_id = str(patient.get("patient_id", "")).strip()
    patient_node_id = f"patient:{patient_id}"

    gender = normalize_gender(patient.get("gender", "female"))

    # 注意：get_patient_view() 返回的是展示层结构，
    # 原始 clinical 未直接暴露，所以这里从 physiology 里反推回统一字段名。
    phys = _safe_dict(patient.get("physiology"))
    core = _safe_dict(phys.get("core_metrics"))
    desc = _safe_dict(phys.get("descriptive_metrics"))

    clinical = {
        "resting_pressure": core.get("肛门括约肌静息压 (mmHg)"),
        "msp": core.get("最大缩榨压 MSP (mmHg)"),
        "squeeze_duration": core.get("缩肛持续时间 (s)"),
        "defecatory_rectal_pressure": core.get("排便时直肠压力 (mmHg)"),
        "first_sensation": desc.get("初始感觉阈值 (ml)"),
        "desire_to_defecate": desc.get("初始便意阈值 (ml)"),
        "urgency_threshold": desc.get("排便窘迫感阈值 (ml)"),
        "max_tolerable_volume": desc.get("最大容量感觉阈值 (ml)"),
        "rair_min_volume": desc.get("RAIR 诱发最小容积 (ml)"),
        "anal_length": desc.get("肛门括约肌长度 (cm)"),
    }

    feature_states = build_patient_feature_states(clinical=clinical, gender=gender)

    def add(fid: str, label: str, value: Any = None, status: str | None = None, source_metric: str | None = None):
        node_id = f"feature:{fid}"
        props = {}
        if value is not None:
            props["value"] = value
        if status:
            props["status"] = status
        if source_metric:
            props["source_metric"] = source_metric

        nodes.append(
            _make_node(
                node_id=node_id,
                label=label,
                node_type="Feature",
                properties=props,
            )
        )
        edges.append(
            _make_edge(
                source=patient_node_id,
                target=node_id,
                relation="HAS_FEATURE",
                properties={"source_type": "clinical_rule"},
            )
        )
        context.append(fid)

    # 静息压
    rp = feature_states.get("resting_pressure")
    rp_value = clinical.get("resting_pressure")
    if rp == "low":
        add("low_resting_pressure", "Low resting pressure", rp_value, rp, "resting_pressure")
    elif rp == "high":
        add("high_resting_pressure", "High resting pressure", rp_value, rp, "resting_pressure")

    # MSP
    msp = feature_states.get("msp")
    msp_value = clinical.get("msp")
    if msp == "low":
        add("low_squeeze_pressure", "Low squeeze pressure", msp_value, msp, "msp")
    elif msp == "high":
        add("high_squeeze_pressure", "High squeeze pressure", msp_value, msp, "msp")

    # 缩肛持续时间
    sqd = feature_states.get("squeeze_duration")
    sqd_value = clinical.get("squeeze_duration")
    if sqd == "low":
        add("short_squeeze_duration", "Short squeeze duration", sqd_value, sqd, "squeeze_duration")
    elif sqd == "high":
        add("long_squeeze_duration", "Long squeeze duration", sqd_value, sqd, "squeeze_duration")

    # 排便时直肠压力
    drp = feature_states.get("defecatory_rectal_pressure")
    drp_value = clinical.get("defecatory_rectal_pressure")
    if drp == "low":
        add("poor_propulsion", "Poor propulsion", drp_value, drp, "defecatory_rectal_pressure")
    elif drp == "high":
        add("high_propulsion", "High rectal propulsive force", drp_value, drp, "defecatory_rectal_pressure")

    # 感觉功能：初始感觉
    fs = feature_states.get("first_sensation")
    fs_value = clinical.get("first_sensation")
    if fs == "high":
        add("rectal_hyposensitivity", "Rectal hyposensitivity", fs_value, fs, "first_sensation")
    elif fs == "low":
        add("rectal_hypersensitivity", "Rectal hypersensitivity", fs_value, fs, "first_sensation")

    # 感觉功能：便意阈值
    ut = feature_states.get("urge_threshold")
    ut_value = clinical.get("desire_to_defecate")
    if ut == "high":
        add("impaired_urge_sensation", "Impaired urge sensation", ut_value, ut, "desire_to_defecate")
    elif ut == "low":
        add("low_urge_threshold", "Low urge threshold", ut_value, ut, "desire_to_defecate")

    # 感觉功能：窘迫感阈值
    dt = feature_states.get("distress_threshold")
    dt_value = clinical.get("urgency_threshold")
    if dt == "high":
        add("impaired_distress_sensation", "Impaired distress sensation", dt_value, dt, "urgency_threshold")
    elif dt == "low":
        add("low_distress_threshold", "Low distress threshold", dt_value, dt, "urgency_threshold")

    # 最大容量
    mtv = feature_states.get("max_tolerable_volume")
    mtv_value = clinical.get("max_tolerable_volume")
    if mtv == "high":
        add("increased_rectal_capacity", "Increased rectal capacity", mtv_value, mtv, "max_tolerable_volume")
    elif mtv == "low":
        add("decreased_rectal_capacity", "Decreased rectal capacity", mtv_value, mtv, "max_tolerable_volume")

    # RAIR
    rair = feature_states.get("rair")
    rair_value = clinical.get("rair_min_volume")
    if rair == "abnormal":
        add("rair_abnormal", "RAIR abnormality", rair_value, rair, "rair_min_volume")
    elif rair == "normal":
        add("rair_present", "RAIR present", rair_value, rair, "rair_min_volume")

    # 肛管长度
    anal_length = feature_states.get("anal_length")
    anal_length_value = clinical.get("anal_length")
    if anal_length == "low":
        add("short_anal_length", "Short anal canal length", anal_length_value, anal_length, "anal_length")
    elif anal_length == "high":
        add("long_anal_length", "Long anal canal length", anal_length_value, anal_length, "anal_length")

    # fallback，避免空图
    if not context:
        add("no_major_abnormality", "No major abnormality", None, "normal", None)

    return nodes, edges, context, feature_states


def build_phenotype_layer(feature_context: List[str]):
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    context: List[str] = []

    def add(pid: str, label: str) -> str:
        node_id = f"phenotype:{pid}"
        nodes.append(_make_node(node_id, label, "Phenotype", {}))
        context.append(pid)
        return node_id

    def connect(fid: str, pid_node_id: str):
        edges.append(
            _make_edge(
                source=f"feature:{fid}",
                target=pid_node_id,
                relation="SUPPORTS_PHENOTYPE",
                properties={"source_type": "rule"},
            )
        )

    if "low_squeeze_pressure" in feature_context:
        pid = add("sphincter_weakness_pattern", "Sphincter weakness pattern")
        connect("low_squeeze_pressure", pid)

    if "poor_propulsion" in feature_context:
        pid = add("poor_propulsion_pattern", "Poor propulsion pattern")
        connect("poor_propulsion", pid)

    if any(x in feature_context for x in [
        "rectal_hyposensitivity",
        "impaired_urge_sensation",
        "impaired_distress_sensation",
        "increased_rectal_capacity",
    ]):
        pid = add("sensory_dysfunction", "Sensory dysfunction")
        for fid in [
            "rectal_hyposensitivity",
            "impaired_urge_sensation",
            "impaired_distress_sensation",
            "increased_rectal_capacity",
        ]:
            if fid in feature_context:
                connect(fid, pid)

    if "rectal_hypersensitivity" in feature_context:
        pid = add("sensory_hyperreactivity", "Sensory hyperreactivity")
        connect("rectal_hypersensitivity", pid)

    if "rair_abnormal" in feature_context:
        pid = add("rair_abnormality", "RAIR abnormality")
        connect("rair_abnormal", pid)

    if "no_major_abnormality" in feature_context:
        pid = add("no_major_phenotype_abnormality", "No major phenotype abnormality")
        connect("no_major_abnormality", pid)

    return nodes, edges, context


def build_mechanism_layer(feature_context: List[str]):
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    context: List[str] = []

    def add(mid: str, label: str) -> str:
        node_id = f"mechanism:{mid}"
        nodes.append(_make_node(node_id, label, "Mechanism", {}))
        context.append(mid)
        return node_id

    def connect(fid: str, mid_node_id: str):
        edges.append(
            _make_edge(
                source=f"feature:{fid}",
                target=mid_node_id,
                relation="INDICATES_MECHANISM",
                properties={"source_type": "rule"},
            )
        )

    if "low_squeeze_pressure" in feature_context:
        mid = add("sphincter_weakness", "Anal sphincter weakness")
        connect("low_squeeze_pressure", mid)

    if "poor_propulsion" in feature_context:
        mid = add("inadequate_propulsion", "Inadequate rectal propulsion")
        connect("poor_propulsion", mid)

    if any(x in feature_context for x in [
        "rectal_hyposensitivity",
        "impaired_urge_sensation",
        "impaired_distress_sensation",
        "increased_rectal_capacity",
    ]):
        mid = add("sensory_impairment", "Rectal sensory impairment")
        for fid in [
            "rectal_hyposensitivity",
            "impaired_urge_sensation",
            "impaired_distress_sensation",
            "increased_rectal_capacity",
        ]:
            if fid in feature_context:
                connect(fid, mid)

    if "rectal_hypersensitivity" in feature_context:
        mid = add("sensory_hyperreactivity", "Rectal sensory hyperreactivity")
        connect("rectal_hypersensitivity", mid)

    if "rair_abnormal" in feature_context:
        mid = add("rair_reflex_abnormality", "RAIR reflex abnormality")
        connect("rair_abnormal", mid)

    return nodes, edges, context


def build_evidence_nodes_from_rag(patient: Dict[str, Any], mechanism_context: List[str]):
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    rag = _safe_dict(patient.get("rag"))
    rag_chunks = rag.get("retrieved_chunks", [])
    if not isinstance(rag_chunks, list):
        rag_chunks = []

    # 最多保留前 4 个证据，避免图过大
    rag_chunks = rag_chunks[:4]

    evidence_node_ids = []
    for item in rag_chunks:
        item = _safe_dict(item)
        chunk_id = str(item.get("chunk_id", "")).strip()
        if not chunk_id:
            continue

        node_id = f"evidence:{chunk_id}"
        nodes.append(
            _make_node(
                node_id=node_id,
                label=item.get("title") or chunk_id,
                node_type="Evidence",
                properties={
                    "chunk_id": chunk_id,
                    "doc_id": item.get("doc_id", ""),
                    "score": item.get("score", 0),
                    "source": item.get("source", ""),
                    "matched_terms": item.get("matched_terms", []),
                    "matched_tags": item.get("matched_tags", []),
                },
            )
        )
        evidence_node_ids.append(node_id)

    # 简化处理：把证据挂到 mechanism；如果没有 mechanism，就挂到 patient
    if mechanism_context and evidence_node_ids:
        for mid in mechanism_context:
            mech_node_id = f"mechanism:{mid}"
            for eid in evidence_node_ids[:3]:
                edges.append(
                    _make_edge(
                        source=mech_node_id,
                        target=eid,
                        relation="HAS_EVIDENCE",
                        properties={"source_type": "rag"},
                    )
                )
    elif evidence_node_ids:
        patient_id = str(patient.get("patient_id", "")).strip()
        patient_node_id = f"patient:{patient_id}"
        for eid in evidence_node_ids[:3]:
            edges.append(
                _make_edge(
                    source=patient_node_id,
                    target=eid,
                    relation="HAS_EVIDENCE",
                    properties={"source_type": "rag"},
                )
            )

    return nodes, edges


def build_explanation_paths(patient_id: str, feature_context: List[str], phenotype_context: List[str], mechanism_context: List[str]):
    paths = []

    # Patient -> Feature -> Phenotype
    for fid in feature_context:
        for pid in phenotype_context:
            paths.append([f"Patient {patient_id}", fid, pid])

    # Patient -> Feature -> Mechanism
    for fid in feature_context:
        for mid in mechanism_context:
            paths.append([f"Patient {patient_id}", fid, mid])

    # 去重
    uniq = []
    seen = set()
    for p in paths:
        key = tuple(p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)

    return uniq[:8]


def get_patient_graph_for_frontend(patient_id: str) -> Dict[str, Any]:
    patient = get_patient_view(patient_id)
    if not patient:
        return {"nodes": [], "edges": [], "paths": []}

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    patient_id = str(patient.get("patient_id", patient_id)).strip()

    # Patient
    nodes.append(
        _make_node(
            node_id=f"patient:{patient_id}",
            label=f"Patient {patient_id}",
            node_type="Patient",
            properties={
                "gender": patient.get("gender", ""),
                "cluster": _safe_dict(patient.get("ai_result")).get("cluster"),
            },
        )
    )

    # Feature
    f_nodes, f_edges, f_ctx, _feature_states = build_observed_feature_nodes(patient)
    nodes.extend(f_nodes)
    edges.extend(f_edges)

    # Phenotype
    p_nodes, p_edges, p_ctx = build_phenotype_layer(f_ctx)
    nodes.extend(p_nodes)
    edges.extend(p_edges)

    # Mechanism
    m_nodes, m_edges, m_ctx = build_mechanism_layer(f_ctx)
    nodes.extend(m_nodes)
    edges.extend(m_edges)

    # Evidence
    e_nodes, e_edges = build_evidence_nodes_from_rag(patient, m_ctx)
    nodes.extend(e_nodes)
    edges.extend(e_edges)

    # Paths
    paths = build_explanation_paths(
        patient_id=patient_id,
        feature_context=f_ctx,
        phenotype_context=p_ctx,
        mechanism_context=m_ctx,
    )

    return {
        "nodes": _dedup_nodes(nodes),
        "edges": _dedup_edges(edges),
        "paths": paths,
    }