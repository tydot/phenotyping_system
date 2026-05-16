from neo4j import GraphDatabase


class Neo4jClient:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
    ):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        if self.driver:
            self.driver.close()

    def run_query(self, query: str, parameters: dict | None = None):
        parameters = parameters or {}
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]