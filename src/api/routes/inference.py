from fastapi import APIRouter, HTTPException
from src.api.schemas.request_models import InferenceRequest, InferenceResponse
from src.ml.inference.engine import InferenceEngine
import asyncio
from functools import partial
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/inference", tags=["inference"])

engine = InferenceEngine()

@router.post("/generate", response_model=InferenceResponse)
async def generate_text(request: InferenceRequest):
    try:
        # Run generation in a separate thread to prevent blocking
        loop = asyncio.get_event_loop()
        generate_func = partial(
            engine.generate,
            request.text,
            max_length=request.max_length,
            temperature=request.temperature
        )
        
        # Set timeout for generation
        generated_text = await asyncio.wait_for(
            loop.run_in_executor(None, generate_func),
            timeout=30.0  # 30 seconds timeout
        )
        
        return InferenceResponse(generated_text=generated_text)
    
    except asyncio.TimeoutError:
        logger.error("Generation timeout")
        raise HTTPException(
            status_code=504,
            detail="Generation timeout. Please try with shorter input or different parameters."
        )
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        ) 