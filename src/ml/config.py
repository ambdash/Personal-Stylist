# Model configurations for fine-tuning and evaluation
# Modern chat LLMs, including Russian-focused models

from dataclasses import dataclass
from typing import Optional, Dict, Any
import os

CACHE_DIR = "/data/dprudnikova/project/models"
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_HOME"] = CACHE_DIR
MAX_LENGTH = 1024

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
    output_dir: str = "models/finetuned"
    
@dataclass
class TestConfig:
    model_name: str
    batch_size: int = 8
    max_length: int = MAX_LENGTH
    num_beams: int = 4
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.2
    metrics: list = ("rouge", "bertscore", "bleu")
    test_size: float = 0.2
    
@dataclass
class InferenceConfig:
    model_name: str
    max_length: int = MAX_LENGTH
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 30
    repetition_penalty: float = 1.2
    do_sample: bool = True

MODEL_CONFIGS = {
    "microsoft/phi-2": {
        "name": "microsoft/phi-2",
        "model_args": {
            "trust_remote_code": True,
            "max_length": MAX_LENGTH,
            "cache_dir": CACHE_DIR
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
            "eval_strategy": "epoch",
            "output_dir": None
        }
    },
    "mistralai/Mistral-7B-v0.1": {
        "name": "mistralai/Mistral-7B-v0.1",
        "model_args": {
            "trust_remote_code": True,
            "max_length": MAX_LENGTH,
            "cache_dir": CACHE_DIR
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
            "output_dir": None
        }
    },
    "IlyaGusev/saiga_llama3_8b": {
        "name": "IlyaGusev/saiga_llama3_8b",
        "model_args": {
            "trust_remote_code": True,
            "max_length": MAX_LENGTH,
            "cache_dir": CACHE_DIR,
            "torch_dtype": "auto",
            "use_cache": False
        },
        "lora_config": {
            "r": 32,
            "lora_alpha": 64,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ],  # Extended target modules for LLaMA
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        },
        "training_args": {
            "num_train_epochs": 5,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 1e-4,
            "fp16": True,
            "logging_steps": 10,
            "save_strategy": "epoch",
            "eval_strategy": "epoch",
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "output_dir": None,
            "gradient_checkpointing": True
        }
    },
    "IlyaGusev/saiga2_7b_lora": {
        "name": "IlyaGusev/saiga2_7b_lora",
        "model_args": {
            "load_in_8bit": True,
            "device_map": "auto",
            "torch_dtype": "auto",
            "max_length": MAX_LENGTH,
            "cache_dir": CACHE_DIR
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
            "load_in_8bit": True,
            "device_map": "auto",
            "torch_dtype": "auto",
            "max_length": MAX_LENGTH,
            "cache_dir": CACHE_DIR
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
    }
} 