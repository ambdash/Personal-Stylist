from neo4j import GraphDatabase
import os
from typing import Any
import logging
from neo4j.exceptions import ServiceUnavailable
import time

logger = logging.getLogger(__name__)

class Neo4jConnection:
    def __init__(self, max_retries=3):
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password123")
        self.max_retries = max_retries
        self.driver = None
        self._connect()
        
    def _connect(self):
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                self.driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.user, self.password),
                    connection_timeout=5,  # 5 seconds connection timeout
                    max_connection_lifetime=3600  # 1 hour max connection lifetime
                )
                # Test the connection
                with self.driver.session() as session:
                    session.run("RETURN 1")
                logger.info(f"Successfully connected to Neo4j at {self.uri}")
                return
            except ServiceUnavailable as e:
                retry_count += 1
                if retry_count == self.max_retries:
                    logger.error(f"Failed to connect to Neo4j after {self.max_retries} attempts: {str(e)}")
                    raise
                logger.warning(f"Connection attempt {retry_count} failed, retrying in 5 seconds...")
                time.sleep(5)
        
    def close(self):
        if self.driver:
            self.driver.close()
        
    def execute_query(self, query: str, parameters: dict = None) -> list[dict[str, Any]]:
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Neo4j query execution failed: {str(e)}")
            raise

    def create_fashion_item(self, item_data: dict):
        query = """
        CREATE (i:FashionItem {
            name: $name,
            category: $category,
            style: $style,
            color: $color,
            season: $season
        })
        RETURN i
        """
        return self.execute_query(query, item_data)

    def create_outfit(self, items: list[str], style: str, occasion: str):
        query = """
        CREATE (o:Outfit {style: $style, occasion: $occasion})
        WITH o
        UNWIND $items as item_name
        MATCH (i:FashionItem {name: item_name})
        CREATE (o)-[:CONTAINS]->(i)
        RETURN o
        """
        return self.execute_query(query, {"items": items, "style": style, "occasion": occasion})

    def get_matching_items(self, style: str, category: str):
        query = """
        MATCH (i:FashionItem)
        WHERE i.style = $style AND i.category = $category
        RETURN i
        """
        return self.execute_query(query, {"style": style, "category": category})

db = Neo4jConnection() 