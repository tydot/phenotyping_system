from __future__ import annotations

from typing import Any, Dict, List

from backend.graph.build_patient_graph import build_patient_knowledge_graph
from backend.graph.neo4j_client import Neo4jClient
from backend.graph.graph_settings import get_neo4j_settings


def safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def sanitize_label(label: str) -> str:
    if not label:
        return "Unknown"
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(label))
    if cleaned and cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    return cleaned or "Unknown"


def upsert_graph_data(
    graph_data: Dict[str, Any],
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
    database: str | None = None,
) -> None:
    cfg = get_neo4j_settings()
    uri = uri or cfg["uri"]
    user = user or cfg["user"]
    password = password or cfg["password"]
    database = database or cfg["database"]

    client = Neo4jClient(
        uri=uri,
        user=user,
        password=password,
        database=database,
    )

    try:
        nodes = safe_list(graph_data.get("nodes"))
        edges = safe_list(graph_data.get("edges"))

        _upsert_nodes(client, nodes)
        _upsert_edges(client, edges)

        print(
            f"Graph upsert completed successfully. "
            f"nodes={len(nodes)}, edges={len(edges)}"
        )
    finally:
        client.close()


def _upsert_nodes(client: Neo4jClient, nodes: List[Dict[str, Any]]) -> None:
    for node in nodes:
        node = safe_dict(node)

        node_id = node.get("id")
        label = node.get("label")
        node_type = sanitize_label(str(node.get("type", "Unknown")))
        properties = safe_dict(node.get("properties"))

        if not node_id or not label:
            continue

        query = f"""
        MERGE (n:{node_type} {{id: $id}})
        SET n.label = $label,
            n.type = $type,
            n += $properties
        RETURN n
        """

        client.run_query(
            query,
            {
                "id": node_id,
                "label": label,
                "type": node_type,
                "properties": properties,
            },
        )


def _upsert_edges(client: Neo4jClient, edges: List[Dict[str, Any]]) -> None:
    query = """
    MATCH (s {id: $source_id})
    MATCH (t {id: $target_id})
    CALL apoc.merge.relationship(
        s,
        $relation,
        {},
        $properties,
        t,
        $properties
    ) YIELD rel
    RETURN rel
    """

    for edge in edges:
        edge = safe_dict(edge)

        source_id = edge.get("source")
        target_id = edge.get("target")
        relation = edge.get("relation")
        properties = safe_dict(edge.get("properties"))

        if not source_id or not target_id or not relation:
            continue

        client.run_query(
            query,
            {
                "source_id": source_id,
                "target_id": target_id,
                "relation": str(relation),
                "properties": properties,
            },
        )


def upsert_patient_knowledge_graph(
    patient: Dict[str, Any],
    patient_id: str | None = None,
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
    database: str | None = None,
) -> Dict[str, Any]:
    graph_data = build_patient_knowledge_graph(patient, patient_id=patient_id)
    upsert_graph_data(
        graph_data=graph_data,
        uri=uri,
        user=user,
        password=password,
        database=database,
    )
    return graph_data