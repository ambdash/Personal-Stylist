from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from typing import Optional, Dict, Any, List
import logging
from functools import lru_cache
import time
from datetime import datetime
from src.ml.config import MODEL_CONFIGS, CACHE_DIR
import os
from transformers import PeftModel

logger = logging.getLogger(__name__)

class InferenceEngine:
    _instance = None
    _models: Dict[str, Any] = {}
    _tokenizers: Dict[str, Any] = {}
    _model_stats: Dict[str, Dict] = {}
    _default_model = "microsoft/phi-2"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InferenceEngine, cls).__new__(cls)
            cls._instance._initialize_stats()
        return cls._instance

    def _initialize_stats(self):
        """Initialize statistics tracking for each model"""
        for model_name in MODEL_CONFIGS:
            self._model_stats[model_name] = {
                "total_requests": 0,
                "total_time": 0.0,
                "errors": 0,
                "last_used": None,
                "loaded": False
            }

    def load_model(self, model_name: str = None, adapter_paths: List[str] = None) -> None:
        """Lazy load model with multiple adapters"""
        if model_name is None:
            model_name = self._default_model
            
        if model_name not in self._models:
            try:
                logger.info(f"Loading model {model_name}")
                start_time = time.time()
                
                # Load base model with 8-bit quantization
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    cache_dir=CACHE_DIR
                )
                
                # Load multiple adapters if provided
                if adapter_paths:
                    for i, adapter_path in enumerate(adapter_paths):
                        if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
                            model = PeftModel.from_pretrained(
                                model,
                                adapter_path,
                                adapter_name=f"adapter_{i}"
                            )
                            logger.info(f"Loaded adapter from {adapter_path}")
                
                self._models[model_name] = model
                
                # Configure tokenizer based on model type
                tokenizer_kwargs = {
                    "trust_remote_code": True,
                    "cache_dir": CACHE_DIR
                }
                
                # Special handling for Saiga and Mistral models
                if any(name in model_name.lower() for name in ["saiga", "mistral"]):
                    tokenizer_kwargs.update({
                        "use_fast": False,
                        "tokenizer_class": "LlamaTokenizer"
                    })
                else:
                    # For other models like Phi-2, use fast tokenizer
                    tokenizer_kwargs.update({
                        "use_fast": True
                    })
                
                # Load tokenizer with appropriate configuration
                self._tokenizers[model_name] = AutoTokenizer.from_pretrained(
                    model_name,
                    **tokenizer_kwargs
                )
                
                if not self._tokenizers[model_name].pad_token_id:
                    self._tokenizers[model_name].pad_token = self._tokenizers[model_name].eos_token
                
                # Update stats
                load_time = time.time() - start_time
                self._model_stats[model_name].update({
                    "loaded": True,
                    "load_time": load_time,
                    "last_load": datetime.now().isoformat()
                })
                
                logger.info(f"Model {model_name} loaded successfully in {load_time:.2f}s")
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {str(e)}")
                self._model_stats[model_name]["errors"] += 1
                raise

    def generate(
        self,
        text: str,
        model_name: Optional[str] = None,
        max_length: int = 1024,
        temperature: float = 0.7
    ) -> tuple[str, float]:
        """Generate text with timing and error handling"""
        if model_name is None:
            model_name = self._default_model
            
        try:
            if model_name not in self._models:
                self.load_model(model_name)

            start_time = time.time()
            
            # Update usage statistics
            self._model_stats[model_name]["last_used"] = datetime.now().isoformat()
            self._model_stats[model_name]["total_requests"] += 1

            # Prepare input
            tokenizer = self._tokenizers[model_name]
            model = self._models[model_name]
            
            # Format input based on model
            if "saiga" in model_name.lower():
                if not text.startswith("Вопрос:"):
                    text = f"Вопрос: {text}"
            
            inputs = tokenizer(
                text, 
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            )

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Generate
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                num_return_sequences=1,
                # do_sample=True,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=temperature > 0
            )

            # Process output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up output based on model
            if "saiga" in model_name.lower():
                # Remove any system prompt or instruction text
                if "Ответ:" in generated_text:
                    generated_text = generated_text.split("Ответ:")[-1].strip()
                # Remove any trailing response markers
                generated_text = generated_text.split("<|end_of_text|>")[0].strip()
            
            generation_time = time.time() - start_time
            
            # Update timing statistics
            self._model_stats[model_name]["total_time"] += generation_time
            
            return generated_text, generation_time

        except Exception as e:
            self._model_stats[model_name]["errors"] += 1
            logger.error(f"Generation error with model {model_name}: {str(e)}")
            raise

    def get_model_stats(self, model_name: Optional[str] = None) -> Dict:
        """Get statistics for one or all models"""
        if model_name:
            stats = self._model_stats.get(model_name, {})
            if stats and stats["total_requests"] > 0:
                stats["avg_inference_time"] = stats["total_time"] / stats["total_requests"]
            return stats
        return self._model_stats

    def health_check(self) -> Dict:
        """Perform health check on all loaded models"""
        health_status = {
            "status": "healthy",
            "models": {}
        }
        
        for model_name in self._models:
            try:
                # Simple inference test
                test_input = "Test input for health check."
                _, gen_time = self.generate(test_input, model_name, max_length=20)
                
                health_status["models"][model_name] = {
                    "status": "healthy",
                    "response_time": gen_time,
                    "loaded": True
                }
            except Exception as e:
                health_status["models"][model_name] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "loaded": False
                }
                health_status["status"] = "degraded"
                
        return health_status 