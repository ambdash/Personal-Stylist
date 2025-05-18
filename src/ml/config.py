# Model configurations for fine-tuning and evaluation
# Modern chat LLMs, including Russian-focused models

from dataclasses import dataclass
from typing import Optional, Dict, Any
import os
import torch
from transformers import BitsAndBytesConfig

# Set up model directories
PROJECT_DATA_DIR = "/data/dprudnikova/project"
CACHE_DIR = os.path.join(PROJECT_DATA_DIR, "models")
TRANSFORMERS_CACHE = os.path.join(PROJECT_DATA_DIR, "models/transformers")
HF_HOME = os.path.join(PROJECT_DATA_DIR, "models/huggingface")

# Create directories if they don't exist
for dir_path in [CACHE_DIR, TRANSFORMERS_CACHE, HF_HOME]:
    os.makedirs(dir_path, exist_ok=True)

# Set environment variables to prevent using .cache
os.environ["TRANSFORMERS_CACHE"] = TRANSFORMERS_CACHE
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_DATASETS_CACHE"] = os.path.join(PROJECT_DATA_DIR, "models/datasets")
os.environ["HF_METRICS_CACHE"] = os.path.join(PROJECT_DATA_DIR, "models/metrics")
os.environ["HF_MODULES_CACHE"] = os.path.join(PROJECT_DATA_DIR, "models/modules")

# Increased max length to accommodate system prompt and longer responses
MAX_LENGTH = 2048

SYSTEM_PROMPT = """Ты — персональный стилист, модный эксперт. Отвечай на вопрос так, как если бы давал совет клиенту. Делай рекомендации точными и стилистически осмысленными. Объясняй, почему тот или иной приём работает. Не пиши очевидного (например, «наденьте топ»). Говори про цвет, настроение, пропорции, фактуры. Учитывай сезон,функциональность и случай, если указано. Пиши по-русски. Ответ должен быть коротким и лаконичным, но содержательным."""

@dataclass
class TrainingConfig:
    model_name: str
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    fp16: bool = True
    system_prompt: str = SYSTEM_PROMPT
    
@dataclass
class TestConfig:
    model_name: str
    batch_size: int = 2
    max_length: int = MAX_LENGTH
    max_new_tokens: int = 512
    num_beams: int = 2
    do_sample: bool = True
    temperature: float = 0.6
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.2
    metrics: list = ("rouge", "bertscore")
    test_size: float = 0.2
    system_prompt: str = SYSTEM_PROMPT
    
@dataclass
class InferenceConfig:
    model_name: str
    max_length: int = MAX_LENGTH
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 30
    repetition_penalty: float = 1.2
    do_sample: bool = True
    system_prompt: str = SYSTEM_PROMPT

# Model download links and configurations
MODEL_DOWNLOAD_LINKS = {
    "mistralai/Mistral-7B-v0.1": {
        "config": "https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/config.json",
        "model": "https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/pytorch_model.bin",
        "tokenizer": "https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/tokenizer.json",
        "tokenizer_config": "https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/tokenizer_config.json",
        "model_index": "https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/model_index.json",
        "generation_config": "https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/generation_config.json"
    },
    "IlyaGusev/saiga2_7b_lora": {
        "config": "https://huggingface.co/IlyaGusev/saiga2_7b_lora/resolve/main/config.json",
        "model": "https://huggingface.co/IlyaGusev/saiga2_7b_lora/resolve/main/pytorch_model.bin",
        "tokenizer": "https://huggingface.co/IlyaGusev/saiga2_7b_lora/resolve/main/tokenizer.json",
        "tokenizer_config": "https://huggingface.co/IlyaGusev/saiga2_7b_lora/resolve/main/tokenizer_config.json",
        "model_index": "https://huggingface.co/IlyaGusev/saiga2_7b_lora/resolve/main/model_index.json",
        "generation_config": "https://huggingface.co/IlyaGusev/saiga2_7b_lora/resolve/main/generation_config.json"
    },
    "IlyaGusev/saiga_mistral_7b": {
        "config": "https://huggingface.co/IlyaGusev/saiga_mistral_7b/resolve/main/config.json",
        "model": "https://huggingface.co/IlyaGusev/saiga_mistral_7b/resolve/main/pytorch_model.bin",
        "tokenizer": "https://huggingface.co/IlyaGusev/saiga_mistral_7b/resolve/main/tokenizer.json",
        "tokenizer_config": "https://huggingface.co/IlyaGusev/saiga_mistral_7b/resolve/main/tokenizer_config.json",
        "model_index": "https://huggingface.co/IlyaGusev/saiga_mistral_7b/resolve/main/model_index.json",
        "generation_config": "https://huggingface.co/IlyaGusev/saiga_mistral_7b/resolve/main/generation_config.json"
    },
    "t-tech/T-lite-it-1.0": {
        "config": "https://huggingface.co/t-tech/T-lite-it-1.0/resolve/main/config.json",
        "model": "https://huggingface.co/t-tech/T-lite-it-1.0/resolve/main/pytorch_model.bin",
        "tokenizer": "https://huggingface.co/t-tech/T-lite-it-1.0/resolve/main/tokenizer.json",
        "tokenizer_config": "https://huggingface.co/t-tech/T-lite-it-1.0/resolve/main/tokenizer_config.json",
        "model_index": "https://huggingface.co/t-tech/T-lite-it-1.0/resolve/main/model_index.json",
        "generation_config": "https://huggingface.co/t-tech/T-lite-it-1.0/resolve/main/generation_config.json"
    }
}

# Create quantization config
def get_quantization_config():
    """Get optimized quantization config for QLoRA."""
    return BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        # load_in_4bit=True,  # Use 4-bit quantization for QLoRA
        # bnb_4bit_use_double_quant=True,
        # bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

MODEL_CONFIGS = {
    "mistralai/Mistral-7B-v0.1": {
        "name": "mistralai/Mistral-7B-v0.1",
        "model_args": {
            "trust_remote_code": True,
            "max_length": MAX_LENGTH,
            "cache_dir": CACHE_DIR,
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
            "quantization_config": get_quantization_config()
        },
        "lora_config": {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        },
        "training_args": {
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "fp16": True,
            "logging_steps": 10,
            "save_strategy": "epoch",
            "eval_strategy": "no",
        }
    },
    "IlyaGusev/saiga2_7b_lora": {
        "name": "IlyaGusev/saiga2_7b_lora",
        "model_args": {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
            "max_length": MAX_LENGTH,
            "cache_dir": CACHE_DIR,
            "quantization_config": get_quantization_config()
        },
        "lora_config": {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        },
        "training_args": {
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "fp16": True,
            "logging_steps": 10,
            "save_strategy": "epoch",
            "evaluation_strategy": "epoch",
            "ddp_find_unused_parameters": False,
            "ddp_backend": "nccl",
            "local_rank": -1
        }
    },
    "IlyaGusev/saiga_mistral_7b": {
        "name": "IlyaGusev/saiga_mistral_7b",
        "model_args": {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
            "max_length": MAX_LENGTH,
            "cache_dir": CACHE_DIR,
            "quantization_config": get_quantization_config()
        },
        "lora_config": {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        },
        "training_args": {
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
        }
    },
    "t-tech/T-lite-it-1.0": {
        "name": "t-tech/T-lite-it-1.0",
        "model_args": {
            "trust_remote_code": True,
            "max_length": MAX_LENGTH,
            "cache_dir": CACHE_DIR,
            # "device_map": None,
            "torch_dtype": torch.float16,
            "quantization_config": get_quantization_config()
        },
        "lora_config": {
            "r": 32,  # Increased rank for better performance
            "lora_alpha": 64,  # Increased alpha for better stability
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Added more target modules
            "lora_dropout": 0.1,  # Increased dropout for better regularization
            "bias": "none",
            "task_type": "CAUSAL_LM"
        },
        "training_args": {
            "num_train_epochs": 1,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 8,  # Increased for better stability
            "learning_rate": 2e-4,
            "fp16": True,  # Disabled fp16 as we're using bf16
            # "bf16": True,  # Using bf16 for better stability
            "logging_steps": 10,
            "save_strategy": "epoch",
            "eval_strategy": "no",
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "gradient_checkpointing": True,
            "optim": "adamw_torch",
            "lr_scheduler_type": "cosine",
            "max_grad_norm": 1.0,
            "dataloader_num_workers": 4,
            "dataloader_pin_memory": True,
            "remove_unused_columns": False,
            "dataloader_num_workers": 0
        }
    }
} 