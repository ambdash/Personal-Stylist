from pydantic import BaseModel
from neo4j import GraphDatabase
import os
from typing import Optional, List, Dict, Any
import logging
import time

logger = logging.getLogger(__name__)

class Neo4jSettings(BaseModel):
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password123"
    database: Optional[str] = None

class Neo4jConfig:
    def __init__(self):
        # Use localhost for local development, neo4j for Docker
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password123")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")
        
        # Connection settings
        self.max_connection_lifetime = int(os.getenv("NEO4J_MAX_CONNECTION_LIFETIME", "3600"))
        self.max_connection_pool_size = int(os.getenv("NEO4J_MAX_CONNECTION_POOL_SIZE", "50"))
        self.connection_acquisition_timeout = int(os.getenv("NEO4J_CONNECTION_ACQUISITION_TIMEOUT", "60"))
        
    def get_connection_params(self) -> dict:
        """Get connection parameters for Neo4j driver"""
        return {
            "uri": self.uri,
            "auth": (self.user, self.password),
            "max_connection_lifetime": self.max_connection_lifetime,
            "max_connection_pool_size": self.max_connection_pool_size,
            "connection_acquisition_timeout": self.connection_acquisition_timeout,
        }

class Neo4jConnection:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password123")
        self._driver = None
        self._max_retries = 5
        self._retry_delay = 3  # seconds
        self._connect()

    def _connect(self):
        """Initialize Neo4j driver with retries"""
        retries = 0
        last_error = None

        while retries < self._max_retries:
            try:
                self._driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.user, self.password)
                )
                # Test connection
                with self._driver.session() as session:
                    session.run("RETURN 1")
                logger.info(f"Successfully connected to Neo4j at {self.uri}")
                return
            except Exception as e:
                last_error = str(e)
                retries += 1
                if retries < self._max_retries:
                    logger.warning(f"Connection attempt {retries} failed, retrying in {self._retry_delay} seconds... Error: {last_error}")
                    time.sleep(self._retry_delay)
                else:
                    logger.error(f"Failed to connect to Neo4j after {self._max_retries} attempts. Last error: {last_error}")
                    raise

    def get_session(self):
        """Get a Neo4j session"""
        if not self._driver:
            self._connect()
        return self._driver.session()

    def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query"""
        try:
            with self.get_session() as session:
                result = session.run(query, parameters or {})
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise

    def close(self):
        """Close the Neo4j connection"""
        if self._driver:
            self._driver.close()
            self._driver = None

# Create a singleton instance
neo4j = Neo4jConnection() 