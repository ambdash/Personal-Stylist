from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging
import os

router = APIRouter(prefix="/telegram/db", tags=["telegram-db"])
logger = logging.getLogger(__name__)

class SearchRequest(BaseModel):
    word: str
    limit: int = 10

class NodeCreateRequest(BaseModel):
    name: str
    node_type: str

class RelationshipCreateRequest(BaseModel):
    from_node_id: str
    to_node_id: str
    relationship_type: str
    properties: Optional[Dict[str, Any]] = None

def get_neo4j_driver():
    """Get Neo4j driver - direct connection"""
    from neo4j import GraphDatabase
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

@router.post("/search")
async def search_nodes_by_word(request: SearchRequest) -> Dict[str, Any]:
    """Search for nodes containing the word - used by Telegram bot"""
    try:
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            # Use regex for word boundary matching to find exact words
            cypher_query = """
            MATCH (n)
            WHERE n.name =~ $search_regex OR n.id =~ $search_regex
            WITH n
            LIMIT $limit
            OPTIONAL MATCH (n)-[r]-(connected)
            WITH n, collect({node: connected, relation: type(r)}) as connections
            RETURN n.name as name, n.id as id, connections,
                   size(connections) as total_connections
            """
            
            # Create regex pattern for word boundary matching
            # (?i) makes it case-insensitive, \\b ensures word boundaries
            search_regex = f"(?i).*\\b{request.word}\\b.*"
            
            result = session.run(cypher_query, search_regex=search_regex, limit=request.limit)
            
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
        
        driver.close()
        
        return {
            "found": len(nodes) > 0,
            "nodes": nodes
        }
        
    except Exception as e:
        logger.error(f"Error searching nodes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/nodes")
async def add_node(request: NodeCreateRequest) -> Dict[str, Any]:
    """Add a new node - used by Telegram bot"""
    try:
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            # Check if node already exists
            check_query = """
            MATCH (n)
            WHERE n.name = $name
            RETURN n.name as name
            """
            existing = session.run(check_query, name=request.name).data()
            
            if existing:
                driver.close()
                return {
                    "success": False,
                    "message": "Node already exists"
                }
            
            # Create new node
            create_query = f"""
            CREATE (n:`{request.node_type}` {{name: $name, id: $name}})
            RETURN n
            """
            
            result = session.run(create_query, name=request.name)
            node = result.single()
            
            if node:
                created_node = dict(node['n'])
                driver.close()
                return {
                    "success": True,
                    "node": created_node
                }
            else:
                driver.close()
                return {
                    "success": False,
                    "message": "Failed to create node"
                }
        
    except Exception as e:
        logger.error(f"Error adding node: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/nodes/{node_id}")
async def delete_node(node_id: str) -> Dict[str, Any]:
    """Delete a node - used by Telegram bot"""
    try:
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            query = """
            MATCH (n {id: $node_id})
            DETACH DELETE n
            RETURN count(n) as deleted_count
            """
            
            result = session.run(query, node_id=node_id)
            record = result.single()
            
            if record and record['deleted_count'] > 0:
                driver.close()
                return {
                    "success": True,
                    "message": f"Node {node_id} deleted successfully"
                }
            else:
                driver.close()
                return {
                    "success": False,
                    "error": f"Node {node_id} not found"
                }
        
    except Exception as e:
        logger.error(f"Error deleting node: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/relationships")
async def create_relationship(request: RelationshipCreateRequest) -> Dict[str, Any]:
    """Create a relationship between nodes - used by Telegram bot"""
    try:
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            if request.properties:
                query = f"""
                MATCH (a {{id: $from_id}}), (b {{id: $to_id}})
                CREATE (a)-[r:{request.relationship_type} $properties]->(b)
                RETURN r
                """
                result = session.run(
                    query,
                    from_id=request.from_node_id,
                    to_id=request.to_node_id,
                    properties=request.properties
                )
            else:
                query = f"""
                MATCH (a {{id: $from_id}}), (b {{id: $to_id}})
                CREATE (a)-[r:{request.relationship_type}]->(b)
                RETURN r
                """
                result = session.run(
                    query,
                    from_id=request.from_node_id,
                    to_id=request.to_node_id
                )
            
            relationship = result.single()
            
            if relationship:
                driver.close()
                return {
                    "success": True,
                    "relationship": dict(relationship['r'])
                }
            else:
                driver.close()
                return {
                    "success": False,
                    "error": "Failed to create relationship"
                }
        
    except Exception as e:
        logger.error(f"Error creating relationship: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/nodes/by-type/{node_type}")
async def get_nodes_by_type(node_type: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Get nodes by type - used by Telegram bot"""
    try:
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            query = f"""
            MATCH (n:`{node_type}`)
            RETURN n.name as name, n.id as id
            LIMIT $limit
            """
            
            result = session.run(query, limit=limit)
            nodes = [{"name": record["name"], "id": record["id"]} for record in result]
        
        driver.close()
        return nodes
        
    except Exception as e:
        logger.error(f"Error getting nodes by type: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check for database operations"""
    try:
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            test_value = result.single()['test']
            
        driver.close()
        
        if test_value == 1:
            return {
                "success": True,
                "message": "Neo4j connection healthy (direct)",
                "method": "direct_connection"
            }
        else:
            return {
                "success": False,
                "error": "Neo4j connection test failed"
            }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 