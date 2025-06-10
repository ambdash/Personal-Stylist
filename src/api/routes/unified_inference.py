from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any, List
from src.ml.config import MODEL_CONFIGS, SYSTEM_PROMPT
from src.celery_app import app as celery_app
import logging
from prometheus_client import Counter, Histogram
import time

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/unified_inference", tags=["unified_inference"])

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

class InferenceParameters(BaseModel):
    """Configurable inference parameters"""
    model_config = ConfigDict(protected_namespaces=())
    
    temperature: float = 0.5      # Less creative, more consistent
    top_p: float = 0.8           # More focused word selection
    top_k: int = 30              # Fewer word options
    repetition_penalty: float = 1.1  # Less aggressive repetition penalty
    max_new_tokens: int = 256        # Shorter responses
    max_length: int = 2048
    do_sample: bool = True
    num_beams: int = 1

class RagInfo(BaseModel):
    """Enhanced RAG information model"""
    model_config = ConfigDict(protected_namespaces=())
    
    enhanced: bool
    strategy_used: Optional[str] = None
    original_prompt: Optional[str] = None
    enhanced_prompt: Optional[str] = None
    concepts_found: int = 0
    concepts: List[Dict[str, Any]] = []
    key_nodes: Dict[str, List[Dict[str, Any]]] = {}
    item_type: Optional[str] = None
    item_subtype: Optional[str] = None
    is_styling: bool = False
    styling_item: Optional[str] = None
    processing_time: float = 0.0
    reason: Optional[str] = None

class UnifiedInferenceRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    prompt: str
    use_rag: bool = False
    model_name: str = "t-tech/T-lite-it-1.0"  # Default to T-lite model
    adapter_path: Optional[str] = None
    system_prompt: Optional[str] = None
    parameters: Optional[InferenceParameters] = None

class UnifiedInferenceResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    generated_text: str
    processing_time: float
    generation_time: float
    model_used: str
    parameters: Dict[str, Any]
    rag_info: Optional[RagInfo] = None
    task_id: str

@router.post("/generate", response_model=UnifiedInferenceResponse)
async def generate_text(request: UnifiedInferenceRequest) -> UnifiedInferenceResponse:
    """
    Generate text using either regular inference or RAG-enhanced inference via Celery
    """
    start_time = time.time()
    inference_type = "rag" if request.use_rag else "regular"
    
    try:
        # Increment request counter
        INFERENCE_REQUESTS.labels(inference_type=inference_type).inc()
        
        # Prepare inference parameters
        params = request.parameters or InferenceParameters()
        
        # Submit task to Celery worker
        task = celery_app.send_task(
            'inference_worker.generate_text',
            kwargs={
                'prompt': request.prompt,
                'use_rag': request.use_rag,
                'model_name': request.model_name,
                'adapter_path': request.adapter_path,
                'system_prompt': request.system_prompt,
                'parameters': params.dict()
            },
            queue='inference'
        )
        
        # Wait for task completion (with timeout)
        try:
            result = task.get(timeout=300)  # 5 minutes timeout
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            raise HTTPException(status_code=500, detail=f"Inference task failed: {str(e)}")
        
        if not result.get('success'):
            raise HTTPException(status_code=500, detail="Inference task failed")
        
        task_result = result['result']
        processing_time = time.time() - start_time
        INFERENCE_LATENCY.labels(inference_type=inference_type).observe(processing_time)

        # Enhanced RAG info processing
        rag_info = None
        if task_result.get("rag_info"):
            raw_rag_info = task_result["rag_info"]
            
            # Determine strategy used based on RAG result
            strategy_used = None
            if raw_rag_info.get("enhanced"):
                if raw_rag_info.get("item_type") or raw_rag_info.get("is_styling"):
                    strategy_used = "item_type_or_styling"
                elif len(raw_rag_info.get("concepts", [])) == 1:
                    strategy_used = "single_keynode_random"
                else:
                    strategy_used = "keynode_intersections"
            
            rag_info = RagInfo(
                enhanced=raw_rag_info.get("enhanced", False),
                strategy_used=strategy_used,
                original_prompt=raw_rag_info.get("original_prompt"),
                enhanced_prompt=raw_rag_info.get("enhanced_prompt"),
                concepts_found=len(raw_rag_info.get("concepts", [])),
                concepts=raw_rag_info.get("concepts", []),
                key_nodes=raw_rag_info.get("key_nodes", {}),
                item_type=raw_rag_info.get("item_type"),
                item_subtype=raw_rag_info.get("item_subtype"),
                is_styling=raw_rag_info.get("is_styling", False),
                styling_item=raw_rag_info.get("styling_item"),
                processing_time=raw_rag_info.get("processing_time", 0.0),
                reason=raw_rag_info.get("reason")
            )

        return UnifiedInferenceResponse(
            generated_text=task_result["generated_text"],
            processing_time=task_result["processing_time"],
            generation_time=task_result["generation_time"],
            model_used=task_result["model_used"],
            parameters=task_result["parameters"],
            rag_info=rag_info,
            task_id=task.id
        )

    except Exception as e:
        INFERENCE_ERRORS.labels(inference_type=inference_type).inc()
        logger.error(f"Unified inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate_async")
async def generate_text_async(request: UnifiedInferenceRequest):
    """
    Submit text generation task asynchronously and return task ID
    """
    try:
        # Prepare inference parameters
        params = request.parameters or InferenceParameters()
        
        # Submit task to Celery worker
        task = celery_app.send_task(
            'inference_worker.generate_text',
            kwargs={
                'prompt': request.prompt,
                'use_rag': request.use_rag,
                'model_name': request.model_name,
                'adapter_path': request.adapter_path,
                'system_prompt': request.system_prompt,
                'parameters': params.dict()
            },
            queue='inference'
        )
        
        return {
            "task_id": task.id,
            "status": "submitted",
            "message": "Task submitted successfully"
        }

    except Exception as e:
        logger.error(f"Failed to submit inference task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/task/{task_id}")
async def get_task_result(task_id: str):
    """
    Get the result of an async task
    """
    try:
        task = celery_app.AsyncResult(task_id)
        
        if task.state == 'PENDING':
            return {
                "task_id": task_id,
                "status": "pending",
                "message": "Task is still processing"
            }
        elif task.state == 'SUCCESS':
            result = task.result
            if result.get('success'):
                return {
                    "task_id": task_id,
                    "status": "completed",
                    "result": result['result']
                }
            else:
                return {
                    "task_id": task_id,
                    "status": "failed",
                    "error": result.get('error', 'Unknown error')
                }
        elif task.state == 'FAILURE':
            return {
                "task_id": task_id,
                "status": "failed",
                "error": str(task.info)
            }
        else:
            return {
                "task_id": task_id,
                "status": task.state.lower(),
                "message": f"Task is in {task.state} state"
            }

    except Exception as e:
        logger.error(f"Failed to get task result: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/status")
async def get_models_status():
    """Get status of all models via Celery task"""
    try:
        task = celery_app.send_task(
            'inference_worker.get_model_stats',
            queue='inference'
        )
        result = task.get(timeout=30)
        
        if result.get('success'):
            return result['stats']
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Failed to get model stats'))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/load/{model_name}")
async def load_model(model_name: str, adapter_path: Optional[str] = None):
    """Explicitly load a model via Celery task"""
    try:
        task = celery_app.send_task(
            'inference_worker.load_model',
            kwargs={
                'model_name': model_name,
                'adapter_path': adapter_path
            },
            queue='inference'
        )
        result = task.get(timeout=120)  # 2 minutes for model loading
        
        if result.get('success'):
            return {
                "status": "success",
                "message": result['message'],
                "model_stats": result.get('model_stats', {})
            }
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Failed to load model'))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/clear_cache")
async def clear_model_cache():
    """Clear model cache via Celery task"""
    try:
        task = celery_app.send_task(
            'inference_worker.clear_cache',
            queue='inference'
        )
        result = task.get(timeout=30)
        
        if result.get('success'):
            return {
                "status": "success",
                "message": result['message']
            }
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Failed to clear cache'))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint via Celery task"""
    try:
        task = celery_app.send_task(
            'inference_worker.health_check',
            queue='inference'
        )
        result = task.get(timeout=30)
        
        if result.get('success'):
            return result['health']
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Health check failed'))
            
    except Exception as e:
        # If Celery is down, return basic health info
        return {
            "status": "degraded",
            "error": f"Celery worker unavailable: {str(e)}",
            "api_status": "healthy"
        } 