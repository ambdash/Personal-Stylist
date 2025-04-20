from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Optional
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

class InferenceEngine:
    _instance = None
    model = None
    tokenizer = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InferenceEngine, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self.model is None:
            self.load_model()

    @lru_cache(maxsize=1)
    def load_model(self):
        """Lazy load model with optimizations"""
        try:
            logger.info("Loading model and tokenizer...")
            model_name = "microsoft/phi-2"
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,  # Use half precision
                low_cpu_mem_usage=True,
                device_map="auto"  # Automatically handle device placement
            )
            
            # Optional: Enable model optimization
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                torch.backends.cudnn.benchmark = True
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def generate(self, text: str, max_length: int = 100, temperature: float = 0.7) -> str:
        """Generate text with timeout and error handling"""
        try:
            # Ensure model is loaded
            if self.model is None:
                self.load_model()

            # Tokenize with max length check
            inputs = self.tokenizer(
                text, 
                return_tensors="pt",
                truncation=True,
                max_length=512  # Prevent too long inputs
            )

            # Move inputs to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Generate with optimized parameters
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                length_penalty=1.0,
                early_stopping=True
            )

            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            raise 