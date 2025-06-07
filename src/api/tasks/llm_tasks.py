from src.celery_app import app as celery_app
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class InferenceConfig:
    """Simple config for inference"""
    model_name: str = "t-tech/T-lite-it-1.0"
    max_length: int = 2048
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 30
    repetition_penalty: float = 1.2
    do_sample: bool = True
    system_prompt: str = """Ты — персональный стилист, модный эксперт. Отвечай на вопрос так, как если бы давал совет клиенту. Делай рекомендации точными и стилистически осмысленными. Объясняй, почему тот или иной приём работает. Не пиши очевидного (например, «наденьте топ»). Говори про цвет, настроение, пропорции, фактуры. Учитывай сезон,функциональность и случай, если указано. Пиши по-русски. Ответ должен быть коротким и лаконичным, но содержательным."""

def get_model_tokenizer(model_name: str):
    """Load model and tokenizer"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

@celery_app.task(name="inference", queue="llm")
def inference(prompt: str, model_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Direct inference with LLM"""
    try:
        config = InferenceConfig(**(model_params or {}))
        model, tokenizer = get_model_tokenizer(config.model_name)
        
        # Prepare prompt with system prompt
        full_prompt = f"{config.system_prompt}\n\nВопрос: {prompt}\n\nОтвет:"
        
        # Generate response
        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=config.max_length)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                do_sample=config.do_sample,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the answer part
        answer = response.split("Ответ:")[-1].strip()
        
        return {
            "operation": "inference",
            "success": True,
            "response": answer
        }
    except Exception as e:
        logger.error(f"Error in inference: {e}")
        return {
            "operation": "inference",
            "success": False,
            "error": str(e)
        }

@celery_app.task(name="rag_inference", queue="llm")
def rag_inference(prompt: str, model_params: Optional[Dict[str, Any]] = None, rag_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """RAG-enhanced inference"""
    try:
        from src.api.db.neo4j.service import Neo4jService
        neo4j_service = Neo4jService()
        
        # Get relevant context from Neo4j
        context = neo4j_service.get_relevant_context(
            prompt,
            max_results=rag_params.get("max_context_chunks", 5) if rag_params else 5,
            similarity_threshold=rag_params.get("similarity_threshold", 0.7) if rag_params else 0.7
        )
        
        # Enhance prompt with context
        enhanced_prompt = f"Контекст: {context}\n\nВопрос: {prompt}"
        
        # Use regular inference with enhanced prompt
        inference_result = inference(enhanced_prompt, model_params)
        
        if inference_result["success"]:
            return {
                "operation": "rag_inference",
                "success": True,
                "response": inference_result["response"],
                "context_used": bool(context)
            }
        else:
            return inference_result
    except Exception as e:
        logger.error(f"Error in RAG inference: {e}")
        return {
            "operation": "rag_inference",
            "success": False,
            "error": str(e)
        } 