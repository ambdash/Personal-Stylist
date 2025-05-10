from pydantic_settings import BaseSettings
from neo4j import GraphDatabase
import os
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class Neo4jSettings(BaseSettings):
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password123"
    database: Optional[str] = None

    class Config:
        env_prefix = "NEO4J_"

class Neo4jConnection:
    def __init__(self, settings: Optional[Neo4jSettings] = None):
        self.settings = settings or Neo4jSettings()
        self._driver = None
        self._connect()

    def _connect(self):
        try:
            self._driver = GraphDatabase.driver(
                self.settings.uri,
                auth=(self.settings.user, self.settings.password)
            )
            # Test connection
            with self._driver.session(database=self.settings.database) as session:
                session.run("RETURN 1")
            logger.info(f"Successfully connected to Neo4j at {self.settings.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise

    def get_session(self):
        return self._driver.session(database=self.settings.database)

    def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return the results"""
        try:
            with self.get_session() as session:
                result = session.run(query, parameters or {})
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise

    def close(self):
        if self._driver:
            self._driver.close()

neo4j_connection = Neo4jConnection() 