from neo4j import GraphDatabase
import os
import logging
from typing import Dict, Any, Optional
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

class Neo4jSettings(BaseSettings):
    uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user: str = os.getenv("NEO4J_USER", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "password123")
    database: Optional[str] = None

    class Config:
        env_prefix = "NEO4J_"

class Neo4jConnection:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Neo4jConnection, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.settings = Neo4jSettings()
            self._driver = None
            self._connect()
            self._initialized = True

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

    def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> list:
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

# Create a singleton instance
neo4j_connection = Neo4jConnection() 