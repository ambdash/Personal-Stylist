from fastapi import APIRouter, HTTPException
from src.api.schemas.request_models import InferenceRequest, InferenceResponse
from src.ml.inference.engine import InferenceEngine
import asyncio
from functools import partial
import logging
from prometheus_client import Counter, Histogram
import time

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/inference", tags=["inference"])

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

engine = InferenceEngine()

@router.post("/generate", response_model=InferenceResponse)
async def generate_text(request: InferenceRequest):
    model_name = request.model_name.value if request.model_name else None
    
    try:
        # Start timing
        start_time = time.time()
        
        # Increment request counter
        INFERENCE_REQUESTS.labels(model_name=model_name or "default").inc()
        
        # Run generation in a separate thread to prevent blocking
        loop = asyncio.get_event_loop()
        generate_func = partial(
            engine.generate,
            request.text,
            model_name=model_name,
            max_length=request.max_length,
            temperature=request.temperature
        )
        
        # Set timeout for generation
        generated_text, generation_time = await asyncio.wait_for(
            loop.run_in_executor(None, generate_func),
            timeout=400.0
        )
        
        # Record latency
        total_time = time.time() - start_time
        INFERENCE_LATENCY.labels(model_name=model_name or "default").observe(total_time)
        
        return InferenceResponse(
            generated_text=generated_text,
            model_used=model_name or "default",
            generation_time=generation_time
        )
    
    except asyncio.TimeoutError:
        INFERENCE_ERRORS.labels(model_name=model_name or "default").inc()
        logger.error("Generation timeout")
        raise HTTPException(
            status_code=504,
            detail="Generation timeout. Please try with shorter input or different parameters."
        )
    except Exception as e:
        INFERENCE_ERRORS.labels(model_name=model_name or "default").inc()
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )

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