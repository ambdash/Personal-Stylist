from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, List
from ..services.rag_service import RagService
from ..db.neo4j.config import Neo4jConnection
from pydantic import BaseModel

router = APIRouter(prefix="/rag", tags=["rag"])

class RagRequest(BaseModel):
    prompt: str

class RagResponse(BaseModel):
    status: str
    extracted_nodes: List[Dict[str, Any]]
    knowledge_chunks: List[List[Dict[str, Any]]]
    prompt_additions: List[str]

def get_neo4j_connection():
    """Get Neo4j connection for dependency injection"""
    return Neo4jConnection()

@router.post("/inference", response_model=RagResponse)
async def rag_inference(
    request: RagRequest,
    rag_service: RagService = Depends(lambda: RagService(get_neo4j_connection()))
) -> Dict[str, Any]:
    """
    Process a prompt through the RAG pipeline.
    
    The pipeline:
    1. Extracts relevant nodes from the prompt
    2. Finds intersections between extracted nodes
    3. Returns structured knowledge from the graph
    4. Provides prompt additions for enhanced context
    """
    try:
        result = await rag_service.process_rag_query(request.prompt)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 