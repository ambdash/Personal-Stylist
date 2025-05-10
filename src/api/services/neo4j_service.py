from typing import List, Dict, Any, Optional
from ..db.neo4j_config import neo4j_connection
import logging

logger = logging.getLogger(__name__)

class Neo4jService:
    @staticmethod
    async def get_graph_structure() -> Dict[str, Any]:
        """Get the complete graph structure"""
        try:
            with neo4j_connection.get_session() as session:
                result = session.run("""
                    CALL db.schema.visualization()
                    YIELD nodes, relationships
                    RETURN nodes, relationships
                """)
                return result.single()
        except Exception as e:
            logger.error(f"Failed to get graph structure: {str(e)}")
            raise

    @staticmethod
    async def add_node(label: str, properties: Dict[str, Any]) -> str:
        """Add a new node with given label and properties"""
        try:
            with neo4j_connection.get_session() as session:
                result = session.run(
                    f"""
                    CREATE (n:{label} $properties)
                    RETURN id(n) as node_id
                    """,
                    properties=properties
                )
                return str(result.single()["node_id"])
        except Exception as e:
            logger.error(f"Failed to add node: {str(e)}")
            raise

    @staticmethod
    async def add_relationship(
        from_label: str,
        from_props: Dict[str, Any],
        to_label: str,
        to_props: Dict[str, Any],
        relationship_type: str,
        rel_props: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a relationship between two nodes"""
        try:
            with neo4j_connection.get_session() as session:
                result = session.run(
                    f"""
                    MATCH (a:{from_label}), (b:{to_label})
                    WHERE a.name = $from_name AND b.name = $to_name
                    CREATE (a)-[r:{relationship_type} $props]->(b)
                    RETURN id(r) as rel_id
                    """,
                    from_name=from_props["name"],
                    to_name=to_props["name"],
                    props=rel_props or {}
                )
                return str(result.single()["rel_id"])
        except Exception as e:
            logger.error(f"Failed to add relationship: {str(e)}")
            raise

    @staticmethod
    async def extract_entities_from_prompt(prompt: str) -> List[Dict[str, Any]]:
        """Extract relevant entities from the prompt and get their relationships"""
        with neo4j_connection.get_session() as session:
            # This is a simplified query - you might want to make it more sophisticated
            result = session.run(
                """
                MATCH (n)
                WHERE n.name CONTAINS $prompt
                OPTIONAL MATCH (n)-[r]->(m)
                RETURN n, r, m
                """,
                prompt=prompt
            )
            return [record for record in result]

    @staticmethod
    async def get_style_recommendations(style: str) -> List[Dict[str, Any]]:
        """Get recommendations for a specific style"""
        with neo4j_connection.get_session() as session:
            result = session.run(
                """
                MATCH (s:Style {name: $style})-[r]->(n)
                RETURN n.name as item, type(r) as relationship
                """,
                style=style
            )
            return [record for record in result] 