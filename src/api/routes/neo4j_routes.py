from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from ..services.neo4j_service import Neo4jService
from pydantic import BaseModel

router = APIRouter()

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