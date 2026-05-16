from __future__ import annotations

from typing import Any, Dict, List, Tuple


def safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def is_meaningful_value(value: Any) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def fmt_value(value: Any) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if value is None:
        return "-"
    return str(value)


class PatientGraphBuilder:
    def __init__(self):
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []
        self.paths: List[List[str]] = []
        self._node_ids = set()
        self._edge_keys = set()
        self._node_labels: Dict[str, str] = {}

    def add_node(self, node_id: str, label: str, node_type: str, **properties):
        if not node_id or node_id in self._node_ids:
            return
        node = {
            "id": node_id,
            "label": label,
            "type": node_type,
            "properties": properties or {},
        }
        self.nodes.append(node)
        self._node_ids.add(node_id)
        self._node_labels[node_id] = label

    def add_edge(self, source_id: str, target_id: str, relation: str, **properties):
        if not source_id or not target_id or not relation:
            return
        edge_key = (source_id, target_id, relation)
        if edge_key in self._edge_keys:
            return
        edge = {
            "source": source_id,
            "target": target_id,
            "relation": relation,
            "properties": properties or {},
        }
        self.edges.append(edge)
        self._edge_keys.add(edge_key)

    def add_path(self, *items: str):
        path = [x for x in items if x]
        if len(path) >= 2:
            self.paths.append(path)

    def get_label(self, node_id: str) -> str:
        return self._node_labels.get(node_id, node_id)

    def build(self) -> Dict[str, Any]:
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "paths": self.paths,
        }


def build_patient_knowledge_graph(
    patient: Dict[str, Any],
    patient_id: str | None = None,
) -> Dict[str, Any]:
    """
    将患者视图数据转换成统一图谱结构，不直接写入 Neo4j。
    允许两种调用方式：
    1. build_patient_knowledge_graph(patient_dict, "210259070")
    2. build_patient_knowledge_graph(patient_dict)  # patient_dict 内需包含 patient_id
    """
    builder = PatientGraphBuilder()

    patient = safe_dict(patient)

    resolved_patient_id = patient_id or patient.get("patient_id") or "unknown"

    ai = safe_dict(patient.get("ai_result"))
    phys = safe_dict(patient.get("physiology"))
    rair = safe_dict(patient.get("rair"))
    rome = safe_dict(patient.get("rome_iv"))
    rag = safe_dict(patient.get("rag"))
    rag_recommendations = safe_list(patient.get("rag_recommendations"))
    llm_analysis = safe_dict(patient.get("llm_analysis"))

    patient_node_id = f"patient:{resolved_patient_id}"
    builder.add_node(
        patient_node_id,
        label=f"患者 {resolved_patient_id}",
        node_type="Patient",
        patient_id=resolved_patient_id,
    )

    phenotype_node_id = _add_ai_phenotype(builder, patient_node_id, ai)
    feature_node_ids = _add_feature_nodes(builder, patient_node_id, phys, rair, rome)
    mechanism_node_ids = _add_mechanism_nodes(builder, feature_node_ids, ai, rome)
    evidence_node_ids = _add_evidence_nodes(builder, mechanism_node_ids, phenotype_node_id, rag)
    recommendation_node_ids = _add_recommendation_nodes(
        builder,
        phenotype_node_id,
        mechanism_node_ids,
        rag_recommendations,
        llm_analysis,
    )

    _build_explanation_paths(
        builder=builder,
        patient_id=resolved_patient_id,
        ai=ai,
        rome=rome,
        phenotype_node_id=phenotype_node_id,
        feature_node_ids=feature_node_ids,
        mechanism_node_ids=mechanism_node_ids,
        evidence_node_ids=evidence_node_ids,
        recommendation_node_ids=recommendation_node_ids,
    )

    return builder.build()


def _add_ai_phenotype(
    builder: PatientGraphBuilder,
    patient_node_id: str,
    ai: Dict[str, Any],
) -> str | None:
    cluster = ai.get("cluster")
    if cluster is None:
        return None

    phenotype_node_id = f"phenotype:cluster_{cluster}"
    confidence = ai.get("confidence")
    is_boundary = bool(ai.get("is_boundary", False))

    builder.add_node(
        phenotype_node_id,
        label=f"Cluster {cluster}",
        node_type="Phenotype",
        cluster_id=cluster,
        confidence=confidence,
        is_boundary=is_boundary,
    )
    builder.add_edge(
        patient_node_id,
        phenotype_node_id,
        "BELONGS_TO",
        confidence=confidence,
        is_boundary=is_boundary,
    )
    return phenotype_node_id


def _add_feature_nodes(
    builder: PatientGraphBuilder,
    patient_node_id: str,
    phys: Dict[str, Any],
    rair: Dict[str, Any],
    rome: Dict[str, Any],
) -> List[str]:
    feature_node_ids: List[str] = []

    core_metrics = safe_dict(phys.get("core_metrics"))
    desc_metrics = safe_dict(phys.get("descriptive_metrics"))

    for category_name, metrics in [
        ("ARM核心指标", core_metrics),
        ("ARM描述性指标", desc_metrics),
    ]:
        for metric_name, metric_value in metrics.items():
            if not is_meaningful_value(metric_value):
                continue

            node_id = f"feature:{category_name}:{metric_name}"
            builder.add_node(
                node_id,
                label=f"{metric_name}={fmt_value(metric_value)}",
                node_type="Feature",
                category=category_name,
                name=metric_name,
                value=metric_value,
            )
            builder.add_edge(patient_node_id, node_id, "HAS_FEATURE", source="physiology")
            feature_node_ids.append(node_id)

    rair_features = safe_dict(rair.get("features"))
    if not (isinstance(rair_features, dict) and rair_features.get("available") is False):
        rair_mapping = {
            "dose_ml": "RAIR剂量",
            "dose_valid": "RAIR剂量有效",
            "event_id": "RAIR事件编号",
            "event_valid": "RAIR事件有效",
            "baseline_pressure": "RAIR基线压力",
            "min_pressure": "RAIR最低压力",
            "relaxation_amplitude": "RAIR松弛幅度",
            "t_min": "RAIR达到最低点时间",
            "recovery_possible": "RAIR可恢复",
            "n_frames": "RAIR帧数",
        }

        for key, display_name in rair_mapping.items():
            value = rair_features.get(key)
            if not is_meaningful_value(value):
                continue

            node_id = f"feature:RAIR:{key}"
            builder.add_node(
                node_id,
                label=f"{display_name}={fmt_value(value)}",
                node_type="Feature",
                category="RAIR",
                name=display_name,
                raw_key=key,
                value=value,
            )
            builder.add_edge(patient_node_id, node_id, "HAS_FEATURE", source="rair")
            feature_node_ids.append(node_id)

    rome_category = rome.get("category")
    if is_meaningful_value(rome_category):
        node_id = "feature:RomeIV:category"
        builder.add_node(
            node_id,
            label=f"Rome IV={rome_category}",
            node_type="Feature",
            category="RomeIV",
            name="Rome IV 分类",
            value=rome_category,
        )
        builder.add_edge(patient_node_id, node_id, "HAS_FEATURE", source="rome_iv")
        feature_node_ids.append(node_id)

    for key, display_name in [
        ("propulsion", "推进力"),
        ("coordination", "协调性"),
        ("ratio_msp_mrp", "MSP/MRP比值"),
    ]:
        value = rome.get(key)
        if not is_meaningful_value(value):
            continue

        node_id = f"feature:RomeIV:{key}"
        builder.add_node(
            node_id,
            label=f"{display_name}={fmt_value(value)}",
            node_type="Feature",
            category="RomeIV",
            name=display_name,
            raw_key=key,
            value=value,
        )
        builder.add_edge(patient_node_id, node_id, "HAS_FEATURE", source="rome_iv")
        feature_node_ids.append(node_id)

    return feature_node_ids


def _add_mechanism_nodes(
    builder: PatientGraphBuilder,
    feature_node_ids: List[str],
    ai: Dict[str, Any],
    rome: Dict[str, Any],
) -> List[str]:
    mechanism_node_ids: List[str] = []

    inferred_mechanisms: List[Tuple[str, str]] = []

    cluster = ai.get("cluster")
    if cluster == 1:
        inferred_mechanisms.append(("mechanism:推进力不足", "推进力不足"))
    elif cluster == 2:
        inferred_mechanisms.append(("mechanism:排便协调障碍", "排便协调障碍"))
    elif cluster == 3:
        inferred_mechanisms.append(("mechanism:反射异常", "反射异常"))
    elif cluster == 4:
        inferred_mechanisms.append(("mechanism:混合型异常", "混合型异常"))

    coordination = rome.get("coordination")
    propulsion = rome.get("propulsion")
    rome_category = rome.get("category")

    if is_meaningful_value(coordination) and ("异常" in str(coordination) or "差" in str(coordination)):
        inferred_mechanisms.append(("mechanism:排便协调障碍", "排便协调障碍"))

    if is_meaningful_value(propulsion) and ("弱" in str(propulsion) or "不足" in str(propulsion)):
        inferred_mechanisms.append(("mechanism:推进力不足", "推进力不足"))

    if is_meaningful_value(rome_category):
        category_text = str(rome_category)
        if "协调" in category_text:
            inferred_mechanisms.append(("mechanism:排便协调障碍", "排便协调障碍"))
        if "推进" in category_text or "排便障碍" in category_text:
            inferred_mechanisms.append(("mechanism:推进力不足", "推进力不足"))

    unique_mechs = []
    seen = set()
    for node_id, label in inferred_mechanisms:
        if node_id not in seen:
            unique_mechs.append((node_id, label))
            seen.add(node_id)

    for node_id, label in unique_mechs:
        builder.add_node(
            node_id,
            label=label,
            node_type="Mechanism",
            name=label,
        )
        mechanism_node_ids.append(node_id)

    for feature_node_id in feature_node_ids:
        feature_text = feature_node_id.lower()

        if (
            "coordination" in feature_text
            or "romeiv:coordination" in feature_text
            or "romeiv:category" in feature_text
        ):
            if "mechanism:排便协调障碍" in mechanism_node_ids:
                builder.add_edge(feature_node_id, "mechanism:排便协调障碍", "INDICATES")

        if (
            "propulsion" in feature_text
            or "msp" in feature_text
        ):
            if "mechanism:推进力不足" in mechanism_node_ids:
                builder.add_edge(feature_node_id, "mechanism:推进力不足", "INDICATES")

        if "rair" in feature_text:
            if "mechanism:反射异常" in mechanism_node_ids:
                builder.add_edge(feature_node_id, "mechanism:反射异常", "INDICATES")
            elif "mechanism:排便协调障碍" in mechanism_node_ids:
                builder.add_edge(feature_node_id, "mechanism:排便协调障碍", "INDICATES")

    return mechanism_node_ids


def _add_evidence_nodes(
    builder: PatientGraphBuilder,
    mechanism_node_ids: List[str],
    phenotype_node_id: str | None,
    rag: Dict[str, Any],
) -> List[str]:
    evidence_node_ids: List[str] = []

    rag_chunks = safe_list(rag.get("retrieved_chunks"))

    for i, chunk in enumerate(rag_chunks, 1):
        chunk = safe_dict(chunk)
        chunk_id = chunk.get("chunk_id") or f"chunk_{i}"
        score = chunk.get("score")
        title = chunk.get("title") or f"文献证据 {i}"
        source = chunk.get("source", "")
        chunk_text = chunk.get("chunk_text", "")

        node_id = f"evidence:{chunk_id}"
        builder.add_node(
            node_id,
            label=title,
            node_type="Evidence",
            chunk_id=chunk_id,
            score=score,
            source=source,
            text=chunk_text,
        )
        evidence_node_ids.append(node_id)

        for mechanism_node_id in mechanism_node_ids:
            builder.add_edge(mechanism_node_id, node_id, "EVIDENCED_BY", score=score)

        if phenotype_node_id:
            builder.add_edge(phenotype_node_id, node_id, "EVIDENCED_BY", score=score)

    return evidence_node_ids


def _add_recommendation_nodes(
    builder: PatientGraphBuilder,
    phenotype_node_id: str | None,
    mechanism_node_ids: List[str],
    rag_recommendations: List[Any],
    llm_analysis: Dict[str, Any],
) -> List[str]:
    recommendation_node_ids: List[str] = []

    all_recommendations: List[str] = []

    for rec in rag_recommendations:
        rec = safe_dict(rec)
        text = rec.get("text") or rec.get("title")
        if is_meaningful_value(text):
            all_recommendations.append(str(text))

    for rec in safe_list(llm_analysis.get("recommendations")):
        if is_meaningful_value(rec):
            all_recommendations.append(str(rec))

    deduped = []
    seen = set()
    for item in all_recommendations:
        if item not in seen:
            deduped.append(item)
            seen.add(item)

    for idx, rec_text in enumerate(deduped, 1):
        node_id = f"recommendation:{idx}"
        builder.add_node(
            node_id,
            label=rec_text[:40],
            node_type="Recommendation",
            content=rec_text,
        )
        recommendation_node_ids.append(node_id)

        if phenotype_node_id:
            builder.add_edge(phenotype_node_id, node_id, "RECOMMENDS")

        for mechanism_node_id in mechanism_node_ids:
            builder.add_edge(mechanism_node_id, node_id, "RECOMMENDS")

    return recommendation_node_ids


def _build_explanation_paths(
    builder: PatientGraphBuilder,
    patient_id: str,
    ai: Dict[str, Any],
    rome: Dict[str, Any],
    phenotype_node_id: str | None,
    feature_node_ids: List[str],
    mechanism_node_ids: List[str],
    evidence_node_ids: List[str],
    recommendation_node_ids: List[str],
):
    patient_label = f"患者 {patient_id}"
    phenotype_label = builder.get_label(phenotype_node_id) if phenotype_node_id else ""

    feature_labels = [builder.get_label(fid) for fid in feature_node_ids[:3]]
    mechanism_labels = [builder.get_label(mid) for mid in mechanism_node_ids[:2]]
    evidence_labels = [builder.get_label(eid) for eid in evidence_node_ids[:2]]
    recommendation_labels = [builder.get_label(rid) for rid in recommendation_node_ids[:2]]

    for feature_label in feature_labels:
        if phenotype_label and mechanism_labels:
            builder.add_path(
                patient_label,
                feature_label,
                phenotype_label,
                mechanism_labels[0],
            )

    if phenotype_label and mechanism_labels and evidence_labels:
        builder.add_path(
            patient_label,
            phenotype_label,
            mechanism_labels[0],
            evidence_labels[0],
        )

    if phenotype_label and recommendation_labels:
        builder.add_path(
            patient_label,
            phenotype_label,
            recommendation_labels[0],
        )

    rome_category = rome.get("category")
    if is_meaningful_value(rome_category) and phenotype_label:
        builder.add_path(
            patient_label,
            f"Rome IV {rome_category}",
            phenotype_label,
        )