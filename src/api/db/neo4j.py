from neo4j import GraphDatabase
import os
from typing import Any
import logging

logger = logging.getLogger(__name__)

class Neo4jConnection:
    def __init__(self):
        uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password123")
        
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
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