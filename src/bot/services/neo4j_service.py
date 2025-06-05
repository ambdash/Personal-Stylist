from neo4j import GraphDatabase
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import random
from src.shared.neo4j_connection import neo4j_connection

logger = logging.getLogger(__name__)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

class Neo4jService:
    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        self.driver.close()

    def get_all_node_labels(self) -> List[str]:
        """Get all available node labels with their counts."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)
                RETURN DISTINCT labels(n)[0] as label, count(*) as count
                ORDER BY label
            """)
            return [(record["label"], record["count"]) for record in result]

    def get_random_nodes_by_label(self, label: str, limit: int = 10, exclude_ids: List[str] = None) -> List[Dict]:
        """Get random nodes of a specific label."""
        exclude_clause = ""
        if exclude_ids:
            exclude_clause = "AND NOT n.id IN $exclude_ids"
            
        with self.driver.session() as session:
            query = f"""
                MATCH (n:`{label}`)
                WHERE n.name IS NOT NULL {exclude_clause}
                WITH n, rand() as r
                ORDER BY r
                LIMIT $limit
                RETURN n.id as id, n.name as name, labels(n)[0] as label
            """
            result = session.run(query, limit=limit, exclude_ids=exclude_ids or [])
            return [{"id": record["id"], 
                    "name": record["name"],
                    "label": record["label"]} for record in result]

    def get_node_relationships(self, node_id: str) -> Dict:
        """Get all relationships for a specific node."""
        with self.driver.session() as session:
            query = """
                MATCH (n {id: $node_id})-[r]-(m)
                RETURN type(r) as relation_type, 
                       m.id as related_id,
                       m.name as related_name,
                       labels(m)[0] as related_label
            """
            result = session.run(query, node_id=node_id)
            relationships = []
            for record in result:
                relationships.append({
                    "type": record["relation_type"],
                    "node": {
                        "id": record["related_id"],
                        "name": record["related_name"],
                        "label": record["related_label"]
                    }
                })
            return relationships

    def search_nodes_by_text(self, text: str, limit: int = 10) -> List[Dict]:
        """Search nodes by text in their name property."""
        with self.driver.session() as session:
            query = """
                MATCH (n)
                WHERE toLower(n.name) CONTAINS toLower($text)
                RETURN n.id as id, n.name as name, labels(n)[0] as label
                LIMIT $limit
            """
            result = session.run(query, text=text, limit=limit)
            return [{"id": record["id"],
                    "name": record["name"],
                    "label": record["label"]} for record in result]

    def get_node_info(self, node_id: str) -> Optional[Dict]:
        """Get detailed information about a specific node."""
        with self.driver.session() as session:
            query = """
                MATCH (n {id: $node_id})
                RETURN n.id as id, n.name as name, labels(n)[0] as label
            """
            result = session.run(query, node_id=node_id)
            record = result.single()
            if record:
                return {
                    "id": record["id"],
                    "name": record["name"],
                    "label": record["label"]
                }
            return None

def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

async def search_nodes_by_word(word: str) -> Dict[str, Any]:
    """
    Search for nodes containing the word and return their connections
    """
    try:
        # Search for nodes containing the word
        result = neo4j_connection.execute_query("""
            MATCH (n)
            WHERE n.name CONTAINS $word OR n.id CONTAINS $word
            WITH n
            LIMIT 10
            OPTIONAL MATCH (n)-[r]-(connected)
            WITH n, collect({node: connected, relation: type(r)}) as connections
            RETURN n.name as name, n.id as id, connections,
                   size((n)-[]-()) as total_connections
        """, {"word": word})
        
        nodes = []
        for record in result:
            node = {
                "name": record["name"],
                "id": record["id"],
                "total_connections": record["total_connections"],
                "connections": [
                    {
                        "name": c["node"]["name"] if c["node"] else None,
                        "relation": c["relation"]
                    }
                    for c in record["connections"]
                    if c["node"] is not None
                ]
            }
            nodes.append(node)
        
        return {
            "found": len(nodes) > 0,
            "nodes": nodes
        }
        
    except Exception as e:
        logger.error(f"Error searching nodes: {e}")
        return {"found": False, "error": str(e)}

async def get_style_recommendations(style: str) -> List[str]:
    """Get style recommendations from Neo4j"""
    try:
        result = neo4j_connection.execute_query("""
            MATCH (s:Style {name: $style})-[r]-(i:Item)
            RETURN i.name as item, type(r) as relation
            LIMIT 10
        """, {"style": style})
        
        recommendations = []
        for record in result:
            recommendations.append(f"{record['item']} ({record['relation']})")
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error getting style recommendations: {e}")
        return []

async def get_node_by_type(node_type: str) -> List[Dict[str, Any]]:
    """Get nodes by type"""
    try:
        result = neo4j_connection.execute_query("""
            MATCH (n)
            WHERE any(label IN labels(n) WHERE label = $type)
            RETURN n.name as name, n.id as id
            LIMIT 20
        """, {"type": node_type})
        
        return [{"name": record["name"], "id": record["id"]} for record in result]
        
    except Exception as e:
        logger.error(f"Error getting nodes by type: {e}")
        return []

async def add_node(name: str, node_type: str) -> Dict[str, Any]:
    """Add a new node with given name and type"""
    try:
        # First check if node exists
        existing = neo4j_connection.execute_query("""
            MATCH (n)
            WHERE n.name = $name
            RETURN n.name as name
        """, {"name": name})
        
        if existing:
            return {"success": False, "message": "Node already exists"}
            
        # Create new node
        result = neo4j_connection.execute_query("""
            CREATE (n:`{type}` {{name: $name, id: $name}})
            RETURN n.name as name, n.id as id
        """.format(type=node_type), {"name": name})
        
        if result:
            return {
                "success": True,
                "node": {
                    "name": result[0]["name"],
                    "id": result[0]["id"]
                }
            }
        return {"success": False, "message": "Failed to create node"}
        
    except Exception as e:
        logger.error(f"Error adding node: {e}")
        return {"success": False, "error": str(e)} 