from src.celery_app import app as celery_app
from src.api.db.neo4j.service import Neo4jService
from src.api.db.neo4j.config import neo4j
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)
neo4j_service = Neo4jService()

@celery_app.task(name="create_node", queue="db")
def create_node(name: str, label: str, aliases: Optional[List[str]] = None, properties: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a new node in Neo4j"""
    try:
        # Merge aliases into properties
        node_props = properties or {}
        if aliases:
            node_props["aliases"] = aliases
        
        result = neo4j_service.create_node(name, label, node_props)
        return {
            "operation": "create",
            "success": True,
            "node": result
        }
    except Exception as e:
        logger.error(f"Error creating node: {e}")
        return {
            "operation": "create",
            "success": False,
            "error": str(e)
        }

@celery_app.task(name="update_node", queue="db")
def update_node(node_id: str, name: Optional[str] = None, aliases: Optional[List[str]] = None, properties: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Update an existing node"""
    try:
        # Merge all updates
        updates = properties or {}
        if name:
            updates["name"] = name
        if aliases is not None:  # Allow empty list to clear aliases
            updates["aliases"] = aliases
        
        result = neo4j_service.update_node(node_id, updates)
        return {
            "operation": "update",
            "success": True,
            "node": result
        }
    except Exception as e:
        logger.error(f"Error updating node: {e}")
        return {
            "operation": "update",
            "success": False,
            "error": str(e)
        }

@celery_app.task(name="delete_node", queue="db")
def delete_node(node_id: str) -> Dict[str, Any]:
    """Delete a node"""
    try:
        success = neo4j_service.delete_node(node_id)
        return {
            "operation": "delete",
            "success": success
        }
    except Exception as e:
        logger.error(f"Error deleting node: {e}")
        return {
            "operation": "delete",
            "success": False,
            "error": str(e)
        }

@celery_app.task(name="create_relation", queue="db")
def create_relation(start_node: str, end_node: str, rel_type: str, properties: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a relationship between nodes"""
    try:
        result = neo4j_service.create_relationship(start_node, end_node, rel_type, properties or {})
        return {
            "operation": "relation",
            "success": True,
            "relation": result
        }
    except Exception as e:
        logger.error(f"Error creating relation: {e}")
        return {
            "operation": "relation",
            "success": False,
            "error": str(e)
        }

@celery_app.task(name="search_nodes", queue="db")
def search_nodes(query: str, label: Optional[str] = None) -> Dict[str, Any]:
    """Search for nodes"""
    try:
        results = neo4j_service.search_nodes(query, label)
        return {
            "operation": "search",
            "success": True,
            "nodes": results
        }
    except Exception as e:
        logger.error(f"Error searching nodes: {e}")
        return {
            "operation": "search",
            "success": False,
            "error": str(e)
        }

@celery_app.task(name="search_by_style", queue="db")
def search_by_style(style: str, limit: int = 10) -> Dict[str, Any]:
    """Search items by style"""
    try:
        query = """
        MATCH (s:Style {name: $style})-[:INCLUDES]->(i:Item)
        RETURN i.name as name, i.type as type
        LIMIT $limit
        """
        result = neo4j.execute_query(query, {"style": style, "limit": limit})
        return {
            "status": "success",
            "items": result
        }
    except Exception as e:
        logger.error(f"Error searching by style: {e}")
        return {"status": "error", "message": str(e)}

@celery_app.task(name="search_by_text", queue="db")
def search_by_text(query: str, label: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
    """Search nodes by text"""
    try:
        # Search in both name and aliases
        cypher_query = """
        MATCH (n)
        WHERE (toLower(n.name) CONTAINS toLower($query) OR 
              ANY(alias IN n.aliases WHERE toLower(alias) CONTAINS toLower($query)))
        AND (CASE WHEN $label IS NULL THEN true ELSE n:`$label` END)
        RETURN n
        LIMIT $limit
        """
        result = neo4j_service.execute_query(
            cypher_query,
            {"query": query, "label": label, "limit": limit}
        )
        return {
            "status": "success",
            "nodes": result
        }
    except Exception as e:
        logger.error(f"Error searching by text: {e}")
        return {
            "status": "error",
            "message": str(e)
        } 