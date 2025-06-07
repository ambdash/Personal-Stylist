import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from peft import PeftModel
import logging
from typing import Tuple, Optional
from pathlib import Path
import os
from huggingface_hub import login

from src.ml.config import MODEL_CONFIGS, CACHE_DIR, TRANSFORMERS_CACHE, HF_HOME

logger = logging.getLogger(__name__)

def get_model_tokenizer(model_name: str, model_path: Optional[str] = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Get model and tokenizer with proper configuration."""
    logger.info(f"Loading model and tokenizer for {model_name}")
    
    # Ensure we're logged in to HuggingFace
    try:
        login(token=os.getenv("HF_TOKEN"), add_to_git_credential=False)
    except Exception as e:
        logger.warning(f"Failed to login to HuggingFace: {e}")
    
    model_config = MODEL_CONFIGS[model_name]
    
    # Check if model exists locally
    local_model_path = Path(CACHE_DIR) / model_name.replace("/", "_")
    if local_model_path.exists():
        logger.info(f"Found local model at {local_model_path}")
        model_name = str(local_model_path)
        # Add local_files_only to prevent downloading
        model_config["model_args"]["local_files_only"] = True
    
    # Configure tokenizer based on model type
    tokenizer_kwargs = {
        "trust_remote_code": True,
        "cache_dir": TRANSFORMERS_CACHE,
        "local_files_only": True if local_model_path.exists() else False,
        "token": os.getenv("HF_TOKEN")  # Add token for authentication
    }
    
    if "saiga" in model_name.lower():
        # Saiga models use LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            **tokenizer_kwargs
        )
    elif "mistral" in model_name.lower():
        # Mistral models need specific configuration
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            padding_side="left",
            **tokenizer_kwargs
        )
    elif "gemma" in model_name.lower():
        # Gemma models need specific configuration
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            padding_side="left",
            **tokenizer_kwargs
        )
    else:
        # Default configuration for other models
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            **tokenizer_kwargs
        )
    
    if not tokenizer.pad_token_id:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Update model config to use project directories and authentication
    model_config["model_args"]["cache_dir"] = TRANSFORMERS_CACHE
    model_config["model_args"]["token"] = os.getenv("HF_TOKEN")
    
    # Load model with quantization
    logger.info(f"Loading model from {model_name} with config: {model_config['model_args']}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_config["model_args"]
    )
    
    # Load finetuned adapter if provided
    if model_path and Path(model_path).exists():
        logger.info(f"Loading finetuned adapter from {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
    
    logger.info(f"Model loaded successfully. Device: {model.device}")
    return model, tokenizer 