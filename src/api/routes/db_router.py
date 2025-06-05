from fastapi import APIRouter, HTTPException
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from src.api.tasks import create_node, search_nodes
from celery.result import AsyncResult

router = APIRouter(prefix="/db", tags=["database"])

# Request/Response Models
class NodeBase(BaseModel):
    label: str
    name: str
    aliases: Optional[List[str]] = None
    properties: Optional[Dict[str, Any]] = None

class RelationBase(BaseModel):
    from_node: str
    to_node: str
    relation_type: str
    properties: Optional[Dict[str, Any]] = None

class SearchQuery(BaseModel):
    query: str
    node_types: Optional[List[str]] = None

class TaskResponse(BaseModel):
    task_id: str
    status: str = "pending"

class NodeResponse(BaseModel):
    node_id: Optional[int] = None
    status: str
    message: Optional[str] = None

@router.post("/nodes", response_model=TaskResponse)
async def add_node(node: NodeBase):
    """
    Add a new node to the database
    Example request:
    {
        "label": "Style",
        "name": "Casual",
        "aliases": ["повседневный", "кэжуал"],
        "properties": {
            "description": "Повседневный стиль одежды",
            "season": "all"
        }
    }
    """
    properties = {"name": node.name}
    if node.aliases:
        properties["aliases"] = node.aliases
    if node.properties:
        properties.update(node.properties)
    
    task = create_node.delay(node.name, node.label, properties)
    return {"task_id": task.id}

@router.post("/search", response_model=TaskResponse)
async def search_nodes_endpoint(query: SearchQuery):
    """
    Search nodes in the database
    Example request:
    {
        "query": "casual",
        "node_types": ["Style", "Item"]
    }
    """
    task = search_nodes.delay(query.query, query.node_types[0] if query.node_types else None)
    return {"task_id": task.id}

@router.get("/tasks/{task_id}", response_model=Any)
async def get_task_result(task_id: str):
    """Get the result of an async database operation"""
    task_result = AsyncResult(task_id)
    if not task_result.ready():
        return {"status": "pending"}
    
    result = task_result.get()
    return result 