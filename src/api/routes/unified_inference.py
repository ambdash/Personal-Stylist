from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from src.ml.config import MODEL_CONFIGS, SYSTEM_PROMPT
from src.ml.inference.engine import InferenceEngine
from src.api.services.rag_service import RagService
from src.api.db.neo4j.config import Neo4jConnection
import logging
from prometheus_client import Counter, Histogram
import time

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/unified_inference", tags=["unified_inference"])

# Initialize inference engine
engine = InferenceEngine()

# Initialize metrics
INFERENCE_REQUESTS = Counter(
    'unified_inference_requests_total',
    'Total number of unified inference requests',
    ['inference_type']  # regular or rag
)
INFERENCE_ERRORS = Counter(
    'unified_inference_errors_total',
    'Total number of unified inference errors',
    ['inference_type']
)
INFERENCE_LATENCY = Histogram(
    'unified_inference_latency_seconds',
    'Unified inference latency in seconds',
    ['inference_type']
)

class UnifiedInferenceRequest(BaseModel):
    prompt: str
    use_rag: bool = False
    model_name: str = "t-tech/T-lite-it-1.0"  # Default to T-lite model
    adapter_path: Optional[str] = None
    system_prompt: Optional[str] = None
    inference_params: Optional[Dict[str, Any]] = None

class UnifiedInferenceResponse(BaseModel):
    generated_text: str
    processing_time: float
    rag_info: Optional[Dict[str, Any]] = None

def get_neo4j_connection():
    """Get Neo4j connection for dependency injection"""
    return Neo4jConnection()

@router.post("/generate", response_model=UnifiedInferenceResponse)
async def generate_text(
    request: UnifiedInferenceRequest,
    rag_service: RagService = Depends(lambda: RagService())
) -> Dict[str, Any]:
    """
    Generate text using either regular inference or RAG-enhanced inference
    """
    start_time = time.time()
    inference_type = "rag" if request.use_rag else "regular"
    
    try:
        # Increment request counter
        INFERENCE_REQUESTS.labels(inference_type=inference_type).inc()
        
        # Prepare final prompt
        final_prompt = request.prompt
        rag_info = None
        
        # If RAG is enabled, enhance the prompt with knowledge from Neo4j
        if request.use_rag:
            try:
                rag_result = await rag_service.process_rag_query(request.prompt)
                if rag_result["status"] == "success" and rag_result["prompt_additions"]:
                    # Add context from RAG to the prompt
                    context = "\n".join(rag_result["prompt_additions"])
                    final_prompt = f"Context:\n{context}\n\nUser Query:\n{request.prompt}"
                rag_info = rag_result
            except Exception as e:
                logger.error(f"RAG processing failed: {e}")
                # Continue with regular inference if RAG fails
                pass

        # Prepare inference parameters
        inference_config = {
            "model_name": request.model_name,
            "adapter_path": request.adapter_path,
            "system_prompt": request.system_prompt or SYSTEM_PROMPT,
            **(request.inference_params or {})
        }

        # Generate text using the model
        generated_text = await engine.generate_async(
            prompt=final_prompt,
            **inference_config
        )

        processing_time = time.time() - start_time
        INFERENCE_LATENCY.labels(inference_type=inference_type).observe(processing_time)

        return UnifiedInferenceResponse(
            generated_text=generated_text,
            processing_time=processing_time,
            rag_info=rag_info
        )

    except Exception as e:
        INFERENCE_ERRORS.labels(inference_type=inference_type).inc()
        logger.error(f"Unified inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/status")
async def get_models_status():
    """Get status of all models"""
    return engine.get_model_stats()

@router.post("/models/load/{model_name}")
async def load_model(model_name: str):
    """Explicitly load a model"""
    try:
        engine.load_model(model_name)
        return {"status": "success", "message": f"Model {model_name} loaded successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {str(e)}"
        ) 