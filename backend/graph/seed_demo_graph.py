from backend.graph.neo4j_client import Neo4jClient


def seed_demo_graph():
    client = Neo4jClient("bolt://localhost:7687", "neo4j", "12345678")

    query = """
    MERGE (p:Patient {patient_id: $patient_id})
    MERGE (f:Feature {name: $feature_name})
      ON CREATE SET f.category = $feature_category, f.value = $feature_value, f.status = $feature_status
    MERGE (ph:Phenotype {name: $phenotype_name})
      ON CREATE SET ph.cluster_id = $cluster_id, ph.description = $phenotype_desc
    MERGE (m:Mechanism {name: $mechanism_name})
      ON CREATE SET m.description = $mechanism_desc
    MERGE (e:Evidence {chunk_id: $chunk_id})
      ON CREATE SET e.title = $evidence_title, e.source = $evidence_source, e.score = $evidence_score, e.text = $evidence_text
    MERGE (r:Recommendation {name: $recommendation_name})
      ON CREATE SET r.content = $recommendation_content

    MERGE (p)-[:HAS_FEATURE]->(f)
    MERGE (p)-[:BELONGS_TO {confidence: $confidence, is_boundary: $is_boundary}]->(ph)
    MERGE (f)-[:SUPPORTS {weight: $support_weight}]->(ph)
    MERGE (f)-[:INDICATES]->(m)
    MERGE (m)-[:EVIDENCED_BY]->(e)
    MERGE (ph)-[:RECOMMENDS]->(r)
    """

    params = {
        "patient_id": "210259070",
        "feature_name": "RAIR松弛幅度降低",
        "feature_category": "RAIR",
        "feature_value": "low",
        "feature_status": "abnormal",
        "phenotype_name": "Cluster 2",
        "cluster_id": 2,
        "phenotype_desc": "以协调异常和反射异常相关特征为主的表型",
        "mechanism_name": "排便协调障碍",
        "mechanism_desc": "提示排便过程中盆底或肛门直肠协调异常",
        "chunk_id": "chunk_001",
        "evidence_title": "RAIR abnormality and defecatory dysfunction",
        "evidence_source": "RAG",
        "evidence_score": 0.91,
        "evidence_text": "RAIR异常与部分功能性排便障碍患者相关。",
        "recommendation_name": "建议结合生物反馈评估",
        "recommendation_content": "建议结合临床进一步评估是否适合生物反馈训练。",
        "confidence": 0.87,
        "is_boundary": False,
        "support_weight": 0.78,
    }

    client.run_query(query, params)
    client.close()


if __name__ == "__main__":
    seed_demo_graph()