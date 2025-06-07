import logging
import torch
import os
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.ml.config import MODEL_CONFIGS, SYSTEM_PROMPT, InferenceConfig
from src.ml.training.trainer import get_model_tokenizer
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
    """Simplified version of the text processor for RAG enhancement"""
    
    def __init__(self):
        # Neo4j connection settings
        self.uri = NEO4J_URI
        self.user = NEO4J_USER
        self.password = NEO4J_PASSWORD
        self.driver = None
        self._connect()

    def _connect(self):
        """Establish connection to Neo4j"""
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info(f"Successfully connected to Neo4j at {self.uri} for RAG processing")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            self.driver = None

    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()

    def process_text_for_rag(self, text: str) -> str:
        """Process text and return enhanced prompt with auxiliary information"""
        if not self.driver:
            logger.warning("Neo4j not available, returning original text")
            return text
            
        try:
            # Import the full processor logic
            from scripts.process_text import EnhancedTextProcessor
            processor = EnhancedTextProcessor()
            
            final_prompt, concepts, processing_time = processor.process_text(text)
            processor.close()
            
            logger.info(f"RAG processing completed in {processing_time:.2f}s, found {len(concepts)} concepts")
            return final_prompt
            
        except Exception as e:
            logger.error(f"Error in RAG processing: {str(e)}")
            return text

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

    def _get_or_load_model(self, model_name: str, adapter_path: Optional[str] = None) -> Tuple[Any, Any]:
        """Get model from cache or load it"""
        # Use default adapter if none specified
        if adapter_path is None:
            adapter_path = DEFAULT_ADAPTER_PATH
            
        cache_key = f"{model_name}:{adapter_path or 'base'}"
        
        if cache_key in self._model_cache:
            logger.info(f"Using cached model: {cache_key}")
            return self._model_cache[cache_key]
        
        logger.info(f"Loading model: {model_name}")
        if adapter_path:
            logger.info(f"Loading with adapter: {adapter_path}")
        
        try:
            # Load base model
            model, tokenizer = get_model_tokenizer(model_name)
            
            # Load adapter if provided (following run.py pattern)
            if adapter_path and Path(adapter_path).exists():
                logger.info(f"Loading LoRA adapter from {adapter_path}...")
                model = PeftModel.from_pretrained(model, adapter_path)
                logger.info(f"LoRA adapter loaded successfully from {adapter_path}")
            elif adapter_path:
                logger.warning(f"Adapter path {adapter_path} does not exist, using base model")
            
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

    def generate(
        self,
        prompt: str,
        model_name: str = DEFAULT_MODEL_NAME,
        adapter_path: Optional[str] = None,
        use_rag: bool = False,
        system_prompt: Optional[str] = None,
        max_length: int = 2048,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repetition_penalty: float = 1.2,
        do_sample: bool = True,
        num_beams: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text with configurable parameters"""
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
                enhanced_prompt = self._text_processor.process_text_for_rag(prompt)
                if enhanced_prompt != prompt:
                    final_prompt = enhanced_prompt
                    rag_info = {
                        "enhanced": True,
                        "original_prompt": prompt,
                        "enhanced_prompt": enhanced_prompt
                    }
                    logger.info("Prompt enhanced with RAG information")
                else:
                    rag_info = {"enhanced": False, "reason": "No relevant context found"}
            
            # Get model and tokenizer
            model, tokenizer = self._get_or_load_model(model_name, adapter_path)
            
            # Prepare full prompt with system prompt
            system_text = system_prompt or SYSTEM_PROMPT
            full_prompt = f"{system_text}\n\nВопрос: {final_prompt}\n\nОтвет:"
            
            # Tokenize input
            inputs = tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            )
            
            # Move to device
            if torch.cuda.is_available():
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate
            generation_start = time.time()
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=do_sample,
                    num_beams=num_beams,
                    pad_token_id=tokenizer.eos_token_id,
                    **kwargs
                )
            
            generation_time = time.time() - generation_start
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the answer part
            if "Ответ:" in response:
                answer = response.split("Ответ:")[-1].strip()
            else:
                answer = response.strip()
            
            total_time = time.time() - start_time
            self._model_stats[cache_key]["total_time"] += total_time
            
            result = {
                "generated_text": answer,
                "processing_time": total_time,
                "generation_time": generation_time,
                "model_used": cache_key,
                "parameters": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "repetition_penalty": repetition_penalty,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                    "num_beams": num_beams
                }
            }
            
            if rag_info:
                result["rag_info"] = rag_info
            
            logger.info(f"Generation completed in {total_time:.2f}s")
            return result
            
        except Exception as e:
            self._model_stats[cache_key]["errors"] += 1
            logger.error(f"Generation error with model {cache_key}: {str(e)}")
            raise

    def load_model(self, model_name: str, adapter_path: Optional[str] = None) -> None:
        """Explicitly load a model"""
        self._get_or_load_model(model_name, adapter_path)

    def get_model_stats(self, model_name: Optional[str] = None) -> Dict:
        """Get statistics for one or all models"""
        if model_name:
            return self._model_stats.get(model_name, {})
        return self._model_stats

    def health_check(self) -> Dict:
        """Health check endpoint"""
        return {
            "status": "healthy",
            "models": self.get_model_stats(),
            "rag_available": self._text_processor.driver is not None,
            "default_adapter": DEFAULT_ADAPTER_PATH,
            "neo4j_connection": f"{NEO4J_URI} as {NEO4J_USER}"
        }

    def clear_cache(self):
        """Clear model cache"""
        self._model_cache.clear()
        logger.info("Model cache cleared")

    def __del__(self):
        """Cleanup on destruction"""
        if self._text_processor:
            self._text_processor.close() 