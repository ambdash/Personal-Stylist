from typing import List, Dict, Any, Set
from src.shared.neo4j_connection import neo4j_connection
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class RagService:
    def __init__(self):
        self.neo4j = neo4j_connection

    async def extract_nodes_from_prompt(self, prompt: str) -> List[Dict[str, Any]]:
        """Extract nodes from prompt that exist in Neo4j"""
        # Split prompt into words and clean them
        words = set(word.strip().lower() for word in prompt.split())
        
        # Find matching nodes
        result = self.neo4j.execute_query("""
            MATCH (n)
            WHERE toLower(n.name) IN $words
            RETURN n.name as name, n.id as id, labels(n)[0] as type
        """, {"words": list(words)})
        
        return result

    async def find_node_intersections(self, node_ids: List[str], min_nodes: int = 2) -> List[Dict[str, Any]]:
        """Find intersections between nodes, reducing min_nodes if no intersections found"""
        while min_nodes > 0:
            query = """
            MATCH (n1)
            WHERE n1.id IN $node_ids
            WITH n1
            MATCH path = (n1)-[*1..2]-(n2)
            WHERE n2.id IN $node_ids AND n1 <> n2
            WITH nodes(path) as nodes, relationships(path) as rels
            WHERE size(apoc.coll.toSet([n in nodes | n.id])) >= $min_nodes
            RETURN nodes, rels
            LIMIT 5
            """
            
            result = self.neo4j.execute_query(query, {
                "node_ids": node_ids,
                "min_nodes": min_nodes
            })
            
            if result:
                return self._format_intersections(result)
            
            min_nodes -= 1
        
        return []

    def _format_intersections(self, neo4j_result: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format Neo4j path results into a more usable structure"""
        formatted_results = []
        
        for record in neo4j_result:
            nodes = record["nodes"]
            rels = record["rels"]
            
            path_info = {
                "nodes": [
                    {
                        "id": node["id"],
                        "name": node["name"],
                        "type": node.get("type") or next(iter(node.labels), None)
                    }
                    for node in nodes
                ],
                "relationships": [
                    {
                        "type": rel.type,
                        "start_node": rel.start_node["name"],
                        "end_node": rel.end_node["name"]
                    }
                    for rel in rels
                ]
            }
            formatted_results.append(path_info)
            
        return formatted_results

    async def process_rag_query(self, prompt: str) -> Dict[str, Any]:
        """Process a RAG query through the pipeline"""
        try:
            # 1. Extract nodes from prompt
            extracted_nodes = await self.extract_nodes_from_prompt(prompt)
            if not extracted_nodes:
                return {
                    "status": "no_nodes_found",
                    "extracted_nodes": [],
                    "knowledge_chunks": [],
                    "prompt_additions": []
                }
            
            # 2. Find intersections between nodes
            node_ids = [node["id"] for node in extracted_nodes]
            intersections = await self.find_node_intersections(node_ids)
            
            # 3. Generate prompt additions based on found knowledge
            prompt_additions = self._generate_prompt_additions(intersections)
            
            return {
                "status": "success",
                "extracted_nodes": extracted_nodes,
                "knowledge_chunks": intersections,
                "prompt_additions": prompt_additions
            }
            
        except Exception as e:
            logger.error(f"Error processing RAG query: {e}")
            return {
                "status": "error",
                "error": str(e),
                "extracted_nodes": [],
                "knowledge_chunks": [],
                "prompt_additions": []
            }

    def _generate_prompt_additions(self, intersections: List[Dict[str, Any]]) -> List[str]:
        """Generate natural language additions to the prompt based on found knowledge"""
        additions = []
        
        for intersection in intersections:
            nodes = intersection["nodes"]
            relationships = intersection["relationships"]
            
            # Create relationship descriptions
            for rel in relationships:
                addition = f"{rel['start_node']} {rel['type'].lower()} {rel['end_node']}"
                additions.append(addition)
        
        return additions 