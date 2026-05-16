import os

def get_neo4j_settings():
    return {
        "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "user": os.getenv("NEO4J_USERNAME", os.getenv("NEO4J_USER", "neo4j")),
        "password": os.getenv("NEO4J_PASSWORD", "12345678"),
        "database": os.getenv("NEO4J_DATABASE", "neo4j"),
    }