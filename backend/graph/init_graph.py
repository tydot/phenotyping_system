from backend.graph.neo4j_client import Neo4jClient


def init_graph():
    client = Neo4jClient(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="12345678",
    )

    queries = [
        "CREATE CONSTRAINT patient_id_unique IF NOT EXISTS FOR (n:Patient) REQUIRE n.patient_id IS UNIQUE",
        "CREATE CONSTRAINT phenotype_name_unique IF NOT EXISTS FOR (n:Phenotype) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT mechanism_name_unique IF NOT EXISTS FOR (n:Mechanism) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT recommendation_name_unique IF NOT EXISTS FOR (n:Recommendation) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT evidence_chunk_id_unique IF NOT EXISTS FOR (n:Evidence) REQUIRE n.chunk_id IS UNIQUE",
        "CREATE INDEX feature_name_index IF NOT EXISTS FOR (n:Feature) ON (n.name)",
    ]

    for q in queries:
        client.run_query(q)

    client.close()
    print("Graph constraints and indexes initialized successfully.")


if __name__ == "__main__":
    init_graph()