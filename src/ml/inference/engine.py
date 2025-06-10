import logging
import torch
import os
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import time
import sys
from pathlib import Path
import gc

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.ml.config import MODEL_CONFIGS, SYSTEM_PROMPT, InferenceConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logger = logging.getLogger(__name__)

# Default paths for fine-tuned model from environment
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "t-tech/T-lite-it-1.0")
DEFAULT_ADAPTER_PATH = os.getenv("ADAPTER_PATH", "src/ml/models/finetuned/t_lite")

# Neo4j connection settings from environment
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

class TextProcessor:
    """Enhanced text processor for RAG enhancement using the new EnhancedRagService"""
    
    def __init__(self):
        self.rag_service = None
        self._initialize_rag_service()

    def _initialize_rag_service(self):
        """Initialize the enhanced RAG service"""
        try:
            from src.api.services.enhanced_rag_service import EnhancedRagService
            self.rag_service = EnhancedRagService()
            logger.info("Successfully initialized EnhancedRagService for RAG processing")
        except Exception as e:
            logger.error(f"Failed to initialize EnhancedRagService: {str(e)}")
            self.rag_service = None

    def close(self):
        """Close connections"""
        if self.rag_service and hasattr(self.rag_service, 'neo4j'):
            self.rag_service.neo4j.close()

    async def process_text_for_rag(self, text: str) -> Dict[str, Any]:
        """Process text and return enhanced prompt with auxiliary information"""
        if not self.rag_service:
            logger.warning("EnhancedRagService not available, returning original text")
            return {
                "enhanced_prompt": text,
                "enhanced": False,
                "reason": "RAG service not available"
            }
            
        try:
            # Use the correct method name from EnhancedRagService
            result = await self.rag_service.enhance_prompt_async(text)
            
            if result["status"] == "success":
                logger.info(f"RAG processing completed in {result['processing_time']:.2f}s, enhanced: {result['enhanced']}")
                return {
                    "enhanced_prompt": result["enhanced_prompt"],
                    "enhanced": result["enhanced"],
                    "concepts": result.get("concepts_used", []),
                    "key_nodes": result.get("key_nodes_found", []),
                    "item_type": result.get("item_type", ""),
                    "item_subtype": result.get("item_subtype", ""),
                    "is_styling": result.get("is_styling", False),
                    "styling_item": result.get("styling_item", ""),
                    "strategy_used": result.get("strategy_used", ""),
                    "processing_time": result["processing_time"]
                }
            else:
                logger.error(f"RAG processing failed: {result.get('error', 'Unknown error')}")
                return {
                    "enhanced_prompt": text,
                    "enhanced": False,
                    "reason": f"RAG processing failed: {result.get('error', 'Unknown error')}"
                }
                
        except Exception as e:
            logger.error(f"Error in RAG processing: {str(e)}")
            return {
                "enhanced_prompt": text,
                "enhanced": False,
                "reason": f"RAG processing error: {str(e)}"
            }

    def process_text_for_rag_sync(self, text: str) -> Dict[str, Any]:
        """Synchronous wrapper for RAG processing"""
        import asyncio
        
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, we need to run in a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.process_text_for_rag(text))
                    return future.result()
            else:
                # If no loop is running, we can run directly
                return loop.run_until_complete(self.process_text_for_rag(text))
        except RuntimeError:
            # No event loop exists, create one
            return asyncio.run(self.process_text_for_rag(text))

class InferenceEngine:
    _instance = None
    _model_cache: Dict[str, Any] = {}
    _model_stats: Dict[str, Dict] = {}
    _text_processor = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InferenceEngine, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the inference engine"""
        self._model_cache = {}
        self._model_stats = {}
        self._text_processor = TextProcessor()
        logger.info("InferenceEngine initialized")

    def _load_model_with_adapter(self, base_model_name: str, adapter_path: str) -> Tuple[Any, Any]:
        """Load model with adapter using standard PyTorch loading (no quantization, no device_map)"""
        logger.info(f"Loading base model: {base_model_name}")
        
        # If adapter path exists and contains tokenizer files, load tokenizer from there
        # Otherwise, load from base model
        tokenizer_path = base_model_name
        if adapter_path and Path(adapter_path).exists():
            # Check if adapter has its own tokenizer
            tokenizer_config_path = Path(adapter_path) / "tokenizer_config.json"
            if tokenizer_config_path.exists():
                tokenizer_path = adapter_path
                logger.info(f"Using tokenizer from adapter path: {adapter_path}")
        
        logger.info(f"Loading tokenizer from {tokenizer_path}...")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            use_fast=False
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Load base model without any device_map or quantization
        logger.info("Loading base model without quantization or device_map...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
            # Removed device_map completely
        )
        
        # Manually move to GPU if available
        if torch.cuda.is_available():
            base_model = base_model.cuda()
            logger.info("Model moved to GPU")
        
        logger.info("✅ Base model loaded successfully")
        
        # Load LoRA adapter
        if adapter_path and Path(adapter_path).exists():
            logger.info(f"Loading LoRA adapter from {adapter_path}...")
            model = PeftModel.from_pretrained(base_model, adapter_path)
            logger.info("✅ LoRA adapter loaded successfully")
        else:
            logger.warning(f"Adapter path not found: {adapter_path}, using base model only")
            model = base_model
        
        model.eval()
        
        # Check memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
            logger.info(f"GPU memory used: {memory_used:.2f} GB")
        
        return model, tokenizer

    def _get_or_load_model(self, model_name: str, adapter_path: Optional[str] = None) -> Tuple[Any, Any]:
        """Get model from cache or load it"""
        # Use default adapter if none specified
        if adapter_path is None:
            adapter_path = DEFAULT_ADAPTER_PATH
            
        cache_key = f"{model_name}:{adapter_path or 'base'}"
        
        if cache_key in self._model_cache:
            logger.info(f"Using cached model: {cache_key}")
            return self._model_cache[cache_key]
        
        try:
            # Load model with adapter using proven configuration
            model, tokenizer = self._load_model_with_adapter(model_name, adapter_path)
            
            # Cache the model
            self._model_cache[cache_key] = (model, tokenizer)
            
            # Initialize stats
            if cache_key not in self._model_stats:
                self._model_stats[cache_key] = {
                    "total_requests": 0,
                    "total_time": 0.0,
                    "errors": 0,
                    "last_used": None,
                    "loaded": True
                }
            
            logger.info(f"Model {cache_key} loaded successfully")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model {cache_key}: {str(e)}")
            raise

    def _format_chat_prompt(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """Format prompt using proper chat template"""
        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT
        
        # Use the proven chat format from our validation
        formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        return formatted_prompt

    def generate(
        self,
        prompt: str,
        model_name: str = DEFAULT_MODEL_NAME,
        adapter_path: Optional[str] = None,
        use_rag: bool = False,
        system_prompt: Optional[str] = None,
        max_length: int = 2048,
        max_new_tokens: int = 256,      # Shorter responses
        temperature: float = 0.5,       # Less creative, more consistent
        top_p: float = 0.8,            # More focused word selection
        top_k: int = 30,               # Fewer word options
        repetition_penalty: float = 1.1,  # Less aggressive repetition penalty
        do_sample: bool = True,        # Keep sampling enabled
        num_beams: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text with configurable parameters using proven working settings"""
        start_time = time.time()
        
        # Use default adapter if none specified
        if adapter_path is None:
            adapter_path = DEFAULT_ADAPTER_PATH
            
        cache_key = f"{model_name}:{adapter_path or 'base'}"
        
        try:
            # Update stats
            if cache_key not in self._model_stats:
                self._model_stats[cache_key] = {
                    "total_requests": 0,
                    "total_time": 0.0,
                    "errors": 0,
                    "last_used": None,
                    "loaded": False
                }
            
            self._model_stats[cache_key]["total_requests"] += 1
            self._model_stats[cache_key]["last_used"] = datetime.now().isoformat()
            
            # Process prompt with RAG if requested
            final_prompt = prompt
            rag_info = None
            
            if use_rag:
                logger.info("Processing prompt with RAG enhancement")
                rag_result = self._text_processor.process_text_for_rag_sync(prompt)
                
                if rag_result["enhanced"]:
                    final_prompt = rag_result["enhanced_prompt"]
                    
                    # Convert key_nodes_found list back to dictionary format
                    key_nodes_dict = {}
                    for node in rag_result.get("key_nodes_found", []):
                        node_type = node.get("type", "Unknown")
                        if node_type not in key_nodes_dict:
                            key_nodes_dict[node_type] = []
                        # Remove the type field from the node before adding to dict
                        node_copy = {k: v for k, v in node.items() if k != "type"}
                        key_nodes_dict[node_type].append(node_copy)
                    
                    rag_info = {
                        "enhanced": True,
                        "original_prompt": prompt,
                        "enhanced_prompt": rag_result["enhanced_prompt"],
                        "concepts": rag_result.get("concepts_used", []),
                        "key_nodes": key_nodes_dict,
                        "item_type": rag_result.get("item_type", ""),
                        "item_subtype": rag_result.get("item_subtype", ""),
                        "is_styling": rag_result.get("is_styling", False),
                        "styling_item": rag_result.get("styling_item", ""),
                        "strategy_used": rag_result.get("strategy_used", ""),
                        "processing_time": rag_result.get("processing_time", 0)
                    }
                    logger.info(f"RAG enhanced prompt with {len(rag_result.get('concepts', []))} concepts using strategy: {rag_result.get('strategy_used', 'unknown')}")
                else:
                    rag_info = {
                        "enhanced": False, 
                        "original_prompt": prompt,
                        "enhanced_prompt": prompt,
                        "reason": rag_result.get("reason", "No relevant context found"),
                        "strategy_used": rag_result.get("strategy_used", "none"),
                        "processing_time": rag_result.get("processing_time", 0)
                    }
                    logger.info(f"RAG processing completed but no enhancement: {rag_info['reason']}")
            
            # Get model and tokenizer
            model, tokenizer = self._get_or_load_model(model_name, adapter_path)
            
            # Format prompt using proper chat template
            formatted_prompt = self._format_chat_prompt(final_prompt, system_prompt)
            
            # Tokenize input
            inputs = tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate with proven working parameters
            generation_start = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,  # Use greedy decoding for stability
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                    top_k=top_k if do_sample else None,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    num_beams=num_beams,
                    early_stopping=True
                )
            
            generation_time = time.time() - generation_start
            
            # Decode response (only the new tokens)
            response = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            processing_time = time.time() - start_time
            
            # Update stats
            self._model_stats[cache_key]["total_time"] += processing_time
            self._model_stats[cache_key]["loaded"] = True
            
            result = {
                "generated_text": response,
                "processing_time": processing_time,
                "generation_time": generation_time,
                "model_used": f"{model_name} + {Path(adapter_path).name if adapter_path else 'base'}",
                "parameters": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "repetition_penalty": repetition_penalty,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                    "num_beams": num_beams
                },
                "rag_info": rag_info,
                "prompt_length": len(formatted_prompt),
                "response_length": len(response)
            }
            
            logger.info(f"✅ Generation completed in {processing_time:.2f}s (generation: {generation_time:.2f}s)")
            return result
            
        except Exception as e:
            self._model_stats[cache_key]["errors"] += 1
            logger.error(f"Generation failed for {cache_key}: {str(e)}")
            raise

    def load_model(self, model_name: str, adapter_path: Optional[str] = None) -> None:
        """Preload a model into cache"""
        self._get_or_load_model(model_name, adapter_path)

    def get_model_stats(self, model_name: Optional[str] = None) -> Dict:
        """Get model statistics"""
        if model_name:
            cache_key = f"{model_name}:{DEFAULT_ADAPTER_PATH}"
            return self._model_stats.get(cache_key, {})
        return self._model_stats

    def health_check(self) -> Dict:
        """Health check for inference engine"""
        return {
            "status": "healthy",
            "models_loaded": len(self._model_cache),
            "total_requests": sum(stats.get("total_requests", 0) for stats in self._model_stats.values()),
            "total_errors": sum(stats.get("errors", 0) for stats in self._model_stats.values()),
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }

    def clear_cache(self):
        """Clear model cache"""
        self._model_cache.clear()
        self._model_stats.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model cache cleared")

    def __del__(self):
        """Cleanup on destruction"""
        if self._text_processor:
            self._text_processor.close() 