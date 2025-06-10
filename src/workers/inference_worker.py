import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.celery_inference_app import app
from src.ml.inference.engine import InferenceEngine

logger = logging.getLogger(__name__)

# Initialize inference engine
engine = InferenceEngine()

@app.task(bind=True, name='inference_worker.generate_text')
def generate_text(
    self,
    prompt: str,
    use_rag: bool = False,
    model_name: str = "t-tech/T-lite-it-1.0",
    adapter_path: Optional[str] = None,
    system_prompt: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate text using the fine-tuned model
    
    Args:
        prompt: Input text prompt
        use_rag: Whether to use RAG enhancement
        model_name: Model name to use
        adapter_path: Path to LoRA adapter (uses default if None)
        system_prompt: Custom system prompt
        parameters: Generation parameters
    
    Returns:
        Dict containing generated text and metadata
    """
    try:
        logger.info(f"Processing inference request: {prompt[:50]}...")
        
        # Default parameters
        default_params = {
            "temperature": 0.5,      # Less creative, more consistent
            "top_p": 0.8,           # More focused word selection
            "top_k": 30,            # Fewer word options
            "repetition_penalty": 1.1,  # Less aggressive repetition penalty
            "max_new_tokens": 256,      # Shorter responses
            "max_length": 2048,
            "do_sample": True,
            "num_beams": 1
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        # Generate text
        result = engine.generate(
            prompt=prompt,
            model_name=model_name,
            adapter_path=adapter_path,
            use_rag=use_rag,
            system_prompt=system_prompt,
            **default_params
        )
        
        logger.info(f"Inference completed successfully in {result['processing_time']:.2f}s")
        return {
            "success": True,
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Inference task failed: {str(e)}")
        # Retry the task
        raise self.retry(exc=e, countdown=60, max_retries=3)

@app.task(bind=True, name='inference_worker.load_model')
def load_model(
    self,
    model_name: str,
    adapter_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Preload a model into cache
    
    Args:
        model_name: Model name to load
        adapter_path: Path to LoRA adapter
    
    Returns:
        Dict with loading status
    """
    try:
        logger.info(f"Loading model: {model_name}")
        if adapter_path:
            logger.info(f"With adapter: {adapter_path}")
        
        engine.load_model(model_name, adapter_path)
        
        return {
            "success": True,
            "message": f"Model {model_name} loaded successfully",
            "model_stats": engine.get_model_stats()
        }
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise self.retry(exc=e, countdown=60, max_retries=2)

@app.task(name='inference_worker.get_model_stats')
def get_model_stats() -> Dict[str, Any]:
    """Get model statistics"""
    try:
        return {
            "success": True,
            "stats": engine.get_model_stats()
        }
    except Exception as e:
        logger.error(f"Failed to get model stats: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@app.task(name='inference_worker.health_check')
def health_check() -> Dict[str, Any]:
    """Health check for inference worker"""
    try:
        health = engine.health_check()
        return {
            "success": True,
            "health": health
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@app.task(name='inference_worker.clear_cache')
def clear_model_cache() -> Dict[str, Any]:
    """Clear model cache"""
    try:
        engine.clear_cache()
        return {
            "success": True,
            "message": "Model cache cleared successfully"
        }
    except Exception as e:
        logger.error(f"Failed to clear cache: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        } 