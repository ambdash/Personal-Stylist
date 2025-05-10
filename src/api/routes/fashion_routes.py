from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from ..db.neo4j import Neo4jService, Neo4jConnection
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/fashion", tags=["fashion"])

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
        return await Neo4jService(Neo4jConnection()).get_graph_structure()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/graph/nodes")
async def create_node(node: NodeCreate):
    """Create a new node"""
    try:
        node_id = await Neo4jService(Neo4jConnection()).create_node(
            node.label, 
            node.properties
        )
        return {"node_id": node_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/graph/relationships")
async def create_relationship(relationship: RelationshipCreate):
    """Create a new relationship"""
    try:
        rel_id = await Neo4jService(Neo4jConnection()).create_relationship(
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

@router.post("/process-prompt")
async def process_prompt(prompt: str):
    """Process a fashion-related prompt and get recommendations"""
    try:
        return await Neo4jService(Neo4jConnection()).process_prompt(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommendations/{style}")
async def get_style_recommendations(style: str):
    """Get recommendations for a specific style"""
    try:
        return await Neo4jService(Neo4jConnection()).get_style_recommendations(style)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 