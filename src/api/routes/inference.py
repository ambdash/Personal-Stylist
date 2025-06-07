from fastapi import APIRouter, HTTPException
from src.api.schemas.request_models import InferenceRequest, InferenceResponse
from src.celery_app import app as celery_app
from celery.result import AsyncResult
import logging
from prometheus_client import Counter, Histogram
import time
from src.ml.inference.engine import InferenceEngine

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/inference", tags=["inference"])

# Initialize inference engine
engine = InferenceEngine()

# Initialize metrics
INFERENCE_REQUESTS = Counter(
    'model_inference_requests_total',
    'Total number of inference requests',
    ['model_name']
)
INFERENCE_ERRORS = Counter(
    'model_inference_errors_total',
    'Total number of inference errors',
    ['model_name']
)
INFERENCE_LATENCY = Histogram(
    'model_inference_latency_seconds',
    'Model inference latency in seconds',
    ['model_name']
)

@router.post("/generate_async")
async def generate_text_async(request: InferenceRequest):
    """
    Asynchronously generate text using the LLM model
    Returns a task_id that can be used to check the status
    """
    try:
        # Increment request counter
        INFERENCE_REQUESTS.labels(model_name=request.model_name or "default").inc()
        
        # For now, return a stub response immediately
        return {
            "task_id": "stub_task_id",
            "status": "completed",
            "result": "This is a stub response for testing purposes."
        }
    
    except Exception as e:
        INFERENCE_ERRORS.labels(model_name=request.model_name or "default").inc()
        logger.error(f"Failed to create inference task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of an async task"""
    try:
        # For now, return a stub response
        return {
            "task_id": task_id,
            "status": "completed",
            "result": "This is a stub response for testing purposes."
        }
    except Exception as e:
        logger.error(f"Error checking task status: {e}")
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