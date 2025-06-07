from typing import List, Dict, Any, Optional
from .config import neo4j
from src.api.utils.triple_extractor import TripleExtractor
import logging
from neo4j import GraphDatabase
import time
import os

logger = logging.getLogger(__name__)

class Neo4jService:
    def __init__(self):
        self.triple_extractor = TripleExtractor()
        self.neo4j = neo4j
        # Use localhost for local development, neo4j for Docker
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password123")
        self.driver = None
        self._max_retries = 3
        self._retry_delay = 2
        self._connect()

    def _connect(self):
        """Establish connection to Neo4j"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info(f"Successfully connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise

    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results"""
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise

    @staticmethod
    def format_path_for_prompt(path: Dict[str, Any]) -> str:
        """Format a Neo4j path for human-readable output"""
        try:
            if not path or not isinstance(path, dict):
                return ""
            
            start_node = path.get("start", {})
            end_node = path.get("end", {})
            relationship = path.get("relationship", {})
            
            if start_node and end_node and relationship:
                return f"{start_node.get('name', '')} -> {relationship.get('type', '')} -> {end_node.get('name', '')}"
            return ""
            
        except Exception as e:
            logger.error(f"Error formatting path: {e}")
            return ""

    def get_nodes_by_type(self, node_type: str) -> List[Dict[str, Any]]:
        """Get all nodes of a specific type"""
        try:
            query = f"""
            MATCH (n:`{node_type}`)
            RETURN n.name as name, n.id as id, labels(n)[0] as label
            ORDER BY n.name
            """
            
            result = self.execute_query(query)
            return result
                
        except Exception as e:
            logger.error(f"Error getting nodes by type: {e}")
            return []

    def create_node(self, name: str, label: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new node"""
        query = f"""
        CREATE (n:`{label}` {{name: $name}})
        SET n += $properties
        RETURN n
        """
        result = self.execute_query(query, {"name": name, "properties": properties})
        return result[0]["n"] if result else None

    def update_node(self, node_id: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing node"""
        query = """
        MATCH (n {id: $node_id})
        SET n += $properties
        RETURN n
        """
        result = self.execute_query(query, {"node_id": node_id, "properties": properties})
        return result[0]["n"] if result else None

    def delete_node(self, node_id: str) -> bool:
        """Delete a node"""
        query = """
        MATCH (n {id: $node_id})
        DETACH DELETE n
        """
        self.execute_query(query, {"node_id": node_id})
        return True

    def create_relationship(self, start_node: str, end_node: str, rel_type: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Create a relationship between nodes"""
        query = f"""
        MATCH (start {{id: $start_node}})
        MATCH (end {{id: $end_node}})
        CREATE (start)-[r:`{rel_type}`]->(end)
        SET r += $properties
        RETURN type(r) as type, r as relationship, 
               start.name as start_name, end.name as end_name
        """
        result = self.execute_query(query, {
            "start_node": start_node,
            "end_node": end_node,
            "properties": properties
        })
        return result[0] if result else None

    def search_nodes(self, query: str, label: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for nodes by text"""
        try:
            # Clean up the query by removing 'example' or similar prefixes
            query = query.lower()
            if query.startswith('example'):
                query = query.replace('example', '').strip()
            if query.startswith('например'):
                query = query.replace('например', '').strip()
            
            cypher_query = """
            MATCH (n)
            WHERE (toLower(n.name) CONTAINS toLower($query))
            AND (CASE 
                WHEN $label IS NULL THEN true 
                ELSE $label IN labels(n)
            END)
            RETURN n.name as name, n.id as id, labels(n) as labels,
                   [(n)-[r]->(m) | {type: type(r), target: m.name}] as outgoing_relations,
                   [(n)<-[r]-(m) | {type: type(r), source: m.name}] as incoming_relations
            LIMIT 10
            """
            logger.info(f"Executing search query: {cypher_query} with params: {{'query': {query}, 'label': {label}}}")
            result = self.execute_query(cypher_query, {"query": query, "label": label})
            logger.info(f"Search results: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in search_nodes: {e}")
            raise

    def get_relevant_context(self, query: str, max_results: int = 5, similarity_threshold: float = 0.7) -> str:
        """Get relevant context for RAG from Neo4j"""
        try:
            # First try to find direct matches
            direct_query = """
            MATCH (n)
            WHERE toLower(n.name) CONTAINS toLower($query)
            WITH n
            OPTIONAL MATCH (n)-[r]-(related)
            RETURN DISTINCT n.name as name, 
                   labels(n) as labels,
                   collect(DISTINCT {
                       type: type(r),
                       node: related.name
                   }) as relationships
            LIMIT $max_results
            """
            
            direct_results = self.execute_query(direct_query, {
                "query": query,
                "max_results": max_results
            })
            
            # Format context
            context_parts = []
            
            # Add direct matches with their relationships
            if direct_results:
                context_parts.append("Прямые совпадения:")
                for result in direct_results:
                    context = f"- {result['name']} ({result['labels'][0]})"
                    if result['relationships']:
                        rel_texts = []
                        for rel in result['relationships']:
                            if rel['type'] and rel['node']:
                                rel_texts.append(f"{rel['type']} -> {rel['node']}")
                        if rel_texts:
                            context += f"\n  Связи: {', '.join(rel_texts)}"
                    context_parts.append(context)
            
            # Try to find style-specific information
            style_query = """
            MATCH (s:Style)-[r:INCLUDES]->(i:Item)
            WHERE toLower(s.name) CONTAINS toLower($query)
            RETURN s.name as style, collect(i.name) as items
            LIMIT 3
            """
            
            style_results = self.execute_query(style_query, {"query": query})
            if style_results:
                context_parts.append("\nИнформация о стилях:")
                for result in style_results:
                    items = result['items'][:5]  # Limit items to 5
                    context_parts.append(f"- Стиль {result['style']}: {', '.join(items)}")
            
            # Try to find occasion-specific information
            occasion_query = """
            MATCH (o:Occasion)<-[r:SUITABLE_FOR]-(i:Item)
            WHERE toLower(o.name) CONTAINS toLower($query)
            RETURN o.name as occasion, collect(i.name) as items
            LIMIT 3
            """
            
            occasion_results = self.execute_query(occasion_query, {"query": query})
            if occasion_results:
                context_parts.append("\nПодходящие случаи:")
                for result in occasion_results:
                    items = result['items'][:5]  # Limit items to 5
                    context_parts.append(f"- Для {result['occasion']}: {', '.join(items)}")
            
            return "\n".join(context_parts) if context_parts else ""
            
        except Exception as e:
            logger.error(f"Error getting relevant context: {e}")
            return ""

    def process_fashion_query(self, query: str) -> Dict[str, Any]:
        """Process a fashion-related query using RAG approach"""
        try:
            # 1. Extract relevant nodes from query
            nodes = self.search_nodes(query)
            
            if not nodes:
                return {
                    "status": "no_nodes_found",
                    "message": "No relevant nodes found in the database."
                }
            
            # 2. Find intersections between nodes
            intersections = []
            if len(nodes) >= 2:
                for i in range(len(nodes)):
                    for j in range(i + 1, len(nodes)):
                        node1, node2 = nodes[i], nodes[j]
                        query = """
                        MATCH (n1 {name: $name1})-[r*1..2]-(n2 {name: $name2})
                        RETURN r
                        """
                        result = self.execute_query(query, {
                            "name1": node1["name"],
                            "name2": node2["name"]
                        })
                        if result:
                            intersections.append({
                                "start": node1["name"],
                                "end": node2["name"],
                                "relationships": result
                            })
            
            return {
                "status": "success",
                "nodes": nodes,
                "intersections": intersections
            }
            
        except Exception as e:
            logger.error(f"Error processing fashion query: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    def close(self):
        """Close the Neo4j connection"""
        if self.driver:
            self.driver.close()
            self.driver = None