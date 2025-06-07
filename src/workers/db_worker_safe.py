import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.celery_db_app import app

logger = logging.getLogger(__name__)

# Neo4j connection settings
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

def get_neo4j_driver():
    """Get Neo4j driver with lazy import to avoid SIGSEGV crashes"""
    try:
        from neo4j import GraphDatabase
        return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    except Exception as e:
        logger.error(f"Failed to create Neo4j driver: {str(e)}")
        raise

@app.task(bind=True, name='db_worker_safe.create_node')
def create_node(
    self,
    node_type: str,
    properties: Dict[str, Any]
) -> Dict[str, Any]:
    """Create a new node in Neo4j"""
    driver = None
    try:
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            # Create the node with properties
            query = f"""
            CREATE (n:{node_type})
            SET n += $properties
            RETURN n.id as id, n.name as name, labels(n) as labels
            """
            
            result = session.run(query, properties=properties)
            record = result.single()
            
            if record:
                return {
                    "success": True,
                    "node": {
                        "id": record["id"],
                        "name": record["name"],
                        "labels": record["labels"]
                    }
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to create node"
                }
                
    except Exception as e:
        logger.error(f"Error creating node: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        if driver:
            driver.close()

@app.task(bind=True, name='db_worker_safe.search_nodes')
def search_nodes(
    self,
    search_query: str,
    node_types: Optional[List[str]] = None,
    limit: int = 50
) -> Dict[str, Any]:
    """Search for nodes by text using word boundary matching"""
    driver = None
    try:
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            # Create regex pattern for word boundary matching
            search_regex = f"(?i)\\b{search_query}\\b"
            
            # Build the search query with regex word boundary matching
            if node_types:
                type_filter = " OR ".join([f"'{nt}' IN labels(n)" for nt in node_types])
                query = f"""
                MATCH (n)
                WHERE ({type_filter}) AND 
                      (n.name =~ $search_regex OR n.id =~ $search_regex)
                RETURN n.id as id, n.name as name, labels(n) as labels
                LIMIT $limit
                """
            else:
                query = """
                MATCH (n)
                WHERE n.name =~ $search_regex OR n.id =~ $search_regex
                RETURN n.id as id, n.name as name, labels(n) as labels
                LIMIT $limit
                """
            
            result = session.run(query, search_regex=search_regex, limit=limit)
            nodes = []
            
            for record in result:
                # Get connections for each node
                conn_query = """
                MATCH (n)-[r]-(connected)
                WHERE n.id = $node_id
                RETURN type(r) as relation_type, connected.name as connected_name
                LIMIT 10
                """
                
                conn_result = session.run(conn_query, node_id=record["id"])
                connections = [
                    {
                        "relation_type": conn_record["relation_type"],
                        "connected_name": conn_record["connected_name"]
                    }
                    for conn_record in conn_result
                ]
                
                nodes.append({
                    "id": record["id"],
                    "name": record["name"],
                    "labels": record["labels"],
                    "total_connections": len(connections),
                    "connections": connections
                })
            
            return {
                "success": True,
                "nodes": nodes,
                "count": len(nodes)
            }
            
    except Exception as e:
        logger.error(f"Error searching nodes: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "nodes": [],
            "count": 0
        }
    finally:
        if driver:
            driver.close()

@app.task(bind=True, name='db_worker_safe.delete_node')
def delete_node(
    self,
    node_id: str
) -> Dict[str, Any]:
    """Delete a node and its relationships"""
    driver = None
    try:
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            # First check if node exists
            check_query = "MATCH (n) WHERE n.id = $node_id RETURN n.name as name"
            check_result = session.run(check_query, node_id=node_id)
            node_record = check_result.single()
            
            if not node_record:
                return {
                    "success": False,
                    "error": f"Node with id '{node_id}' not found"
                }
            
            # Delete the node and all its relationships
            delete_query = """
            MATCH (n) WHERE n.id = $node_id
            DETACH DELETE n
            RETURN count(n) as deleted_count
            """
            
            result = session.run(delete_query, node_id=node_id)
            record = result.single()
            
            if record and record["deleted_count"] > 0:
                return {
                    "success": True,
                    "message": f"Node '{node_record['name']}' deleted successfully"
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to delete node"
                }
                
    except Exception as e:
        logger.error(f"Error deleting node: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        if driver:
            driver.close()

@app.task(bind=True, name='db_worker_safe.create_relationship')
def create_relationship(
    self,
    from_node_id: str,
    to_node_id: str,
    relationship_type: str,
    properties: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a relationship between two nodes"""
    driver = None
    try:
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            # Check if both nodes exist
            check_query = """
            MATCH (from_node) WHERE from_node.id = $from_id
            MATCH (to_node) WHERE to_node.id = $to_id
            RETURN from_node.name as from_name, to_node.name as to_name
            """
            
            check_result = session.run(check_query, from_id=from_node_id, to_id=to_node_id)
            check_record = check_result.single()
            
            if not check_record:
                return {
                    "success": False,
                    "error": "One or both nodes not found"
                }
            
            # Create the relationship
            if properties:
                create_query = f"""
                MATCH (from_node) WHERE from_node.id = $from_id
                MATCH (to_node) WHERE to_node.id = $to_id
                CREATE (from_node)-[r:{relationship_type}]->(to_node)
                SET r += $properties
                RETURN type(r) as rel_type
                """
                result = session.run(create_query, 
                                   from_id=from_node_id, 
                                   to_id=to_node_id, 
                                   properties=properties)
            else:
                create_query = f"""
                MATCH (from_node) WHERE from_node.id = $from_id
                MATCH (to_node) WHERE to_node.id = $to_id
                CREATE (from_node)-[r:{relationship_type}]->(to_node)
                RETURN type(r) as rel_type
                """
                result = session.run(create_query, 
                                   from_id=from_node_id, 
                                   to_id=to_node_id)
            
            record = result.single()
            
            if record:
                return {
                    "success": True,
                    "relationship": {
                        "from_node": check_record["from_name"],
                        "to_node": check_record["to_name"],
                        "type": record["rel_type"]
                    }
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to create relationship"
                }
                
    except Exception as e:
        logger.error(f"Error creating relationship: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        if driver:
            driver.close()

@app.task(name='db_worker_safe.health_check')
def health_check() -> Dict[str, Any]:
    """Health check for database worker"""
    driver = None
    try:
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            record = result.single()
            
            if record and record["test"] == 1:
                return {
                    "success": True,
                    "message": "Database worker is healthy",
                    "worker": "db_worker_safe"
                }
            else:
                return {
                    "success": False,
                    "error": "Database connection test failed"
                }
                
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        if driver:
            driver.close() 