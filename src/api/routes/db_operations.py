from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional, List
from ..tasks import db_tasks
from ..schemas.db_schemas import (
    NodeCreate,
    NodeUpdate,
    RelationCreate,
    SearchQuery
)
import logging

router = APIRouter(prefix="/db", tags=["database"])
logger = logging.getLogger(__name__)

@router.post("/nodes/create")
async def create_node(node: NodeCreate):
    """Create a new node in Neo4j
    
    Example:
    ```
    {
        "name": "Белые кроссовки Nike",
        "label": "Концепт",
        "aliases": ["Nike white sneakers", "Найк белые"],
        "properties": {
            "description": "Классические белые кроссовки"
        }
    }
    ```
    """
    try:
        task = db_tasks.create_node.delay(
            name=node.name,
            label=node.label,
            aliases=node.aliases,
            properties=node.properties
        )
        return {"task_id": task.id}
    except Exception as e:
        logger.error(f"Error creating node: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/nodes/{node_id}")
async def update_node(node_id: str, node: NodeUpdate):
    """Update an existing node
    
    Example:
    ```
    {
        "name": "Обновленное название",
        "aliases": ["Новый алиас 1", "Новый алиас 2"],
        "properties": {
            "description": "Обновленное описание"
        }
    }
    ```
    """
    try:
        task = db_tasks.update_node.delay(
            node_id=node_id,
            name=node.name,
            aliases=node.aliases,
            properties=node.properties
        )
        return {"task_id": task.id}
    except Exception as e:
        logger.error(f"Error updating node: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/nodes/{node_id}")
async def delete_node(node_id: str):
    """Delete a node by ID
    
    Example: /nodes/Концепт:белые-кроссовки
    """
    try:
        task = db_tasks.delete_node.delay(node_id=node_id)
        return {"task_id": task.id}
    except Exception as e:
        logger.error(f"Error deleting node: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/relations")
async def create_relation(relation: RelationCreate):
    """Create a relationship between nodes
    
    Example:
    ```
    {
        "start_node": "Концепт:белые-кроссовки",
        "end_node": "Эстетика:minimalism",
        "rel_type": "ОТНОСИТСЯ_К",
        "properties": {
            "confidence": 0.9
        }
    }
    ```
    """
    try:
        task = db_tasks.create_relation.delay(
            start_node=relation.start_node,
            end_node=relation.end_node,
            rel_type=relation.rel_type,
            properties=relation.properties
        )
        return {"task_id": task.id}
    except Exception as e:
        logger.error(f"Error creating relation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search")
async def search_nodes(query: str, label: Optional[str] = None):
    """Search for nodes by text
    
    Examples:
    - /search?query=кроссовки
    - /search?query=минимализм&label=Эстетика
    """
    try:
        task = db_tasks.search_nodes.delay(
            query=query,
            label=label
        )
        return {"task_id": task.id}
    except Exception as e:
        logger.error(f"Error searching nodes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a task"""
    try:
        task = db_tasks.celery_app.AsyncResult(task_id)
        if task.ready():
            if task.successful():
                return {
                    "status": "completed",
                    "result": task.result
                }
            else:
                return {
                    "status": "failed",
                    "error": str(task.result)
                }
        return {"status": "pending"}
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 