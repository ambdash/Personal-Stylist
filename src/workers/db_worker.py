import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.celery_db_app import app

logger = logging.getLogger(__name__)

# Neo4j connection settings from environment
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

def get_neo4j_driver():
    """Get Neo4j driver with environment configuration"""
    from neo4j import GraphDatabase
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

@app.task(bind=True, name='db_worker.create_node')
def create_node(
    self,
    node_type: str,
    properties: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a new node in Neo4j
    
    Args:
        node_type: Type/label of the node
        properties: Node properties
    
    Returns:
        Dict with creation result
    """
    try:
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            # Create node query
            query = f"""
            CREATE (n:{node_type} $properties)
            RETURN n
            """
            
            result = session.run(query, properties=properties)
            node = result.single()
            
            if node:
                created_node = dict(node['n'])
                logger.info(f"Created {node_type} node: {created_node.get('name', 'unnamed')}")
                return {
                    "success": True,
                    "node": created_node
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to create node"
                }
                
    except Exception as e:
        logger.error(f"Failed to create node: {str(e)}")
        raise self.retry(exc=e, countdown=30, max_retries=3)
    finally:
        if 'driver' in locals():
            driver.close()

@app.task(bind=True, name='db_worker.update_node')
def update_node(
    self,
    node_id: str,
    properties: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update an existing node in Neo4j
    
    Args:
        node_id: ID of the node to update
        properties: Properties to update
    
    Returns:
        Dict with update result
    """
    try:
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            # Update node query
            query = """
            MATCH (n {id: $node_id})
            SET n += $properties
            RETURN n
            """
            
            result = session.run(query, node_id=node_id, properties=properties)
            node = result.single()
            
            if node:
                updated_node = dict(node['n'])
                logger.info(f"Updated node {node_id}")
                return {
                    "success": True,
                    "node": updated_node
                }
            else:
                return {
                    "success": False,
                    "error": f"Node {node_id} not found"
                }
                
    except Exception as e:
        logger.error(f"Failed to update node: {str(e)}")
        raise self.retry(exc=e, countdown=30, max_retries=3)
    finally:
        if 'driver' in locals():
            driver.close()

@app.task(bind=True, name='db_worker.delete_node')
def delete_node(
    self,
    node_id: str
) -> Dict[str, Any]:
    """
    Delete a node from Neo4j
    
    Args:
        node_id: ID of the node to delete
    
    Returns:
        Dict with deletion result
    """
    try:
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            # Delete node query
            query = """
            MATCH (n {id: $node_id})
            DETACH DELETE n
            RETURN count(n) as deleted_count
            """
            
            result = session.run(query, node_id=node_id)
            record = result.single()
            
            if record and record['deleted_count'] > 0:
                logger.info(f"Deleted node {node_id}")
                return {
                    "success": True,
                    "message": f"Node {node_id} deleted successfully"
                }
            else:
                return {
                    "success": False,
                    "error": f"Node {node_id} not found"
                }
                
    except Exception as e:
        logger.error(f"Failed to delete node: {str(e)}")
        raise self.retry(exc=e, countdown=30, max_retries=3)
    finally:
        if 'driver' in locals():
            driver.close()

@app.task(bind=True, name='db_worker.create_relationship')
def create_relationship(
    self,
    from_node_id: str,
    to_node_id: str,
    relationship_type: str,
    properties: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a relationship between two nodes
    
    Args:
        from_node_id: ID of the source node
        to_node_id: ID of the target node
        relationship_type: Type of relationship
        properties: Optional relationship properties
    
    Returns:
        Dict with creation result
    """
    try:
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            # Create relationship query
            if properties:
                query = f"""
                MATCH (a {{id: $from_id}}), (b {{id: $to_id}})
                CREATE (a)-[r:{relationship_type} $properties]->(b)
                RETURN r
                """
                result = session.run(
                    query,
                    from_id=from_node_id,
                    to_id=to_node_id,
                    properties=properties
                )
            else:
                query = f"""
                MATCH (a {{id: $from_id}}), (b {{id: $to_id}})
                CREATE (a)-[r:{relationship_type}]->(b)
                RETURN r
                """
                result = session.run(
                    query,
                    from_id=from_node_id,
                    to_id=to_node_id
                )
            
            relationship = result.single()
            
            if relationship:
                logger.info(f"Created {relationship_type} relationship: {from_node_id} -> {to_node_id}")
                return {
                    "success": True,
                    "relationship": dict(relationship['r'])
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to create relationship"
                }
                
    except Exception as e:
        logger.error(f"Failed to create relationship: {str(e)}")
        raise self.retry(exc=e, countdown=30, max_retries=3)
    finally:
        if 'driver' in locals():
            driver.close()

@app.task(bind=True, name='db_worker.search_nodes')
def search_nodes(
    self,
    query: str,
    node_types: Optional[List[str]] = None,
    limit: int = 50
) -> Dict[str, Any]:
    """
    Search for nodes in Neo4j with connections information
    
    Args:
        query: Search query
        node_types: Optional list of node types to filter
        limit: Maximum number of results
    
    Returns:
        Dict with search results including connections
    """
    try:
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            # Build search query with connections
            if node_types:
                labels = ":".join(node_types)
                cypher_query = f"""
                MATCH (n:{labels})
                WHERE n.name CONTAINS $search_term OR any(alias IN n.aliases WHERE alias CONTAINS $search_term)
                WITH n
                LIMIT $limit
                OPTIONAL MATCH (n)-[r]-(connected)
                WITH n, collect({{node: connected, relation: type(r)}}) as connections
                RETURN n.name as name, n.id as id, connections,
                       size(connections) as total_connections
                """
            else:
                cypher_query = """
                MATCH (n)
                WHERE n.name CONTAINS $search_term OR n.id CONTAINS $search_term
                WITH n
                LIMIT $limit
                OPTIONAL MATCH (n)-[r]-(connected)
                WITH n, collect({node: connected, relation: type(r)}) as connections
                RETURN n.name as name, n.id as id, connections,
                       size(connections) as total_connections
                """
            
            result = session.run(cypher_query, search_term=query, limit=limit)
            
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
            
            logger.info(f"Found {len(nodes)} nodes for query: {query}")
            return {
                "success": True,
                "nodes": nodes,
                "count": len(nodes)
            }
                
    except Exception as e:
        logger.error(f"Failed to search nodes: {str(e)}")
        raise self.retry(exc=e, countdown=30, max_retries=3)
    finally:
        if 'driver' in locals():
            driver.close()

@app.task(name='db_worker.health_check')
def health_check() -> Dict[str, Any]:
    """Health check for database worker"""
    try:
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            test_value = result.single()['test']
            
            if test_value == 1:
                return {
                    "success": True,
                    "message": "Neo4j connection healthy",
                    "connection": f"{NEO4J_URI} as {NEO4J_USER}"
                }
            else:
                return {
                    "success": False,
                    "error": "Neo4j connection test failed"
                }
                
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        if 'driver' in locals():
            driver.close() 