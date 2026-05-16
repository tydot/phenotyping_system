from backend.graph.neo4j_client import Neo4jClient


def get_patient_subgraph(patient_id: str):
    client = Neo4jClient("bolt://localhost:7687", "neo4j", "12345678")

    query = """
    MATCH (p:Patient {patient_id: $patient_id})-[r]-(n)
    RETURN p, r, n
    """

    result = client.run_query(query, {"patient_id": patient_id})
    client.close()
    return result