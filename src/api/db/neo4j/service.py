from typing import List, Dict, Any, Optional
from .config import Neo4jConnection
from .queries import Neo4jQueries
import logging
from ...utils.triple_extractor import TripleExtractor

logger = logging.getLogger(__name__)

class Neo4jService:
    def __init__(self, connection: Neo4jConnection):
        self.connection = connection
        self.queries = Neo4jQueries()
        self.triple_extractor = TripleExtractor()

    async def get_graph_structure(self) -> Dict[str, Any]:
        """Get the complete graph structure"""
        with self.connection.get_session() as session:
            result = session.run(self.queries.GET_GRAPH_STRUCTURE)
            return result.single()

    async def create_node(self, label: str, properties: Dict[str, Any]) -> str:
        """Create a new node with given label and properties"""
        with self.connection.get_session() as session:
            result = session.run(
                self.queries.CREATE_NODE,
                label=label,
                properties=properties
            )
            return str(result.single()["node_id"])

    async def create_relationship(
        self,
        from_label: str,
        from_props: Dict[str, Any],
        to_label: str,
        to_props: Dict[str, Any],
        relationship_type: str,
        rel_props: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a relationship between two nodes"""
        with self.connection.get_session() as session:
            result = session.run(
                self.queries.CREATE_RELATIONSHIP,
                from_label=from_label,
                to_label=to_label,
                from_name=from_props["name"],
                to_name=to_props["name"],
                relationship_type=relationship_type,
                props=rel_props or {}
            )
            return str(result.single()["rel_id"])

    async def process_prompt(self, prompt: str) -> Dict[str, Any]:
        """Process a prompt and extract relevant fashion information"""
        # First extract triples from the prompt
        triples = self.triple_extractor.extract_triples(prompt)
        
        # Then get additional context from the database
        with self.connection.get_session() as session:
            result = session.run(
                self.queries.EXTRACT_ENTITIES,
                prompt=prompt
            )
            db_context = [record for record in result]
            
        return {
            "extracted_triples": triples,
            "database_context": db_context
        }

    async def get_style_recommendations(self, style: str) -> List[Dict[str, Any]]:
        """Get comprehensive style recommendations"""
        with self.connection.get_session() as session:
            result = session.run(
                self.queries.GET_STYLE_RECOMMENDATIONS,
                style=style
            )
            return [record for record in result] 