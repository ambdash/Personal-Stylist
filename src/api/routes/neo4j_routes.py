from fastapi import APIRouter, HTTPException, Query
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from typing import Dict, Any, List, Optional
from ..services.neo4j_service import Neo4jService
from pydantic import BaseModel
from ..db.neo4j_config import neo4j_connection
from src.celery_app import app as celery_app
import logging

router = APIRouter(prefix="/neo4j", tags=["neo4j"])
logger = logging.getLogger(__name__)

class NodeCreate(BaseModel):
    label: str
    properties: Dict[str, Any]

class RelationshipCreate(BaseModel):
    from_label: str
    from_properties: Dict[str, Any]
    to_label: str
    to_properties: Dict[str, Any]
    relationship_type: str
    relationship_properties: Dict[str, Any] = {}

@router.get("/graph/structure")
async def get_graph_structure():
    """Get the complete graph structure"""
    try:
        return await Neo4jService.get_graph_structure()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/graph/nodes")
async def create_node(node: NodeCreate):
    """Create a new node"""
    try:
        node_id = await Neo4jService.add_node(node.label, node.properties)
        return {"node_id": node_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/graph/relationships")
async def create_relationship(relationship: RelationshipCreate):
    """Create a new relationship"""
    try:
        rel_id = await Neo4jService.add_relationship(
            relationship.from_label,
            relationship.from_properties,
            relationship.to_label,
            relationship.to_properties,
            relationship.relationship_type,
            relationship.relationship_properties
        )
        return {"relationship_id": rel_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/graph/extract/{prompt}")
async def extract_from_prompt(prompt: str):
    """Extract entities and relationships from a prompt"""
    try:
        return await Neo4jService.extract_entities_from_prompt(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/graph/recommendations/{style}")
async def get_style_recommendations(style: str):
    """Get recommendations for a specific style"""
    try:
        return await Neo4jService.get_style_recommendations(style)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/styles")
async def add_style(name: str, description: Optional[str] = None):
    """Add a new style to Neo4j"""
    try:
        task = celery_app.send_task(
            'add_style_to_neo4j',
            args=[name, description]
        )
        return {"task_id": task.id, "status": "processing"}
    except Exception as e:
        logger.error(f"Error adding style: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/items")
async def add_item(name: str, style_name: str, item_type: Optional[str] = None):
    """Add a new item to Neo4j"""
    try:
        task = celery_app.send_task(
            'add_item_to_neo4j',
            args=[name, style_name, item_type]
        )
        return {"task_id": task.id, "status": "processing"}
    except Exception as e:
        logger.error(f"Error adding item: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/styles")
async def get_styles():
    """Get all styles from Neo4j"""
    try:
        with neo4j_connection.get_session() as session:
            result = session.run("MATCH (s:Style) RETURN s.name as name, s.description as description")
            return [{"name": record["name"], "description": record["description"]} for record in result]
    except Exception as e:
        logger.error(f"Error getting styles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/items/{style_name}")
async def get_items_by_style(style_name: str):
    """Get all items for a specific style"""
    try:
        with neo4j_connection.get_session() as session:
            result = session.run(
                """
                MATCH (s:Style {name: $style_name})<-[:BELONGS_TO]-(i:Item)
                RETURN i.name as name, i.type as type
                """,
                style_name=style_name
            )
            return [{"name": record["name"], "type": record["type"]} for record in result]
    except Exception as e:
        logger.error(f"Error getting items: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 