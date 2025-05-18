import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datasets import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from src.ml.config import MODEL_CONFIGS, SYSTEM_PROMPT

logger = logging.getLogger(__name__)

def format_prompt(example: Dict[str, str], model_type: str = "t-lite") -> Dict[str, str]:
    """Format prompt according to model requirements."""
    if model_type == "t-lite":
        return {
            "text": f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
                   f"<|im_start|>user\n{example['instruction']}\n<|im_end|>\n"
                   f"<|im_start|>assistant\n{example['output']}<|im_end|>"
        }
    else:  # For Saiga models
        return {
            "text": f"<s>system\n{SYSTEM_PROMPT}\n</s>\n"
                   f"<s>user\n{example['instruction']}\n</s>\n"
                   f"<s>assistant\n{example['output']}</s>"
        }

def tokenize_function(examples: Dict[str, Any], tokenizer: Any, max_length: int = 768) -> Dict[str, Any]:
    assert isinstance(examples["text"], list)

    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors=None
    )

    # Добавляем labels — они должны быть такими же, как input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


def get_model_tokenizer(model_name: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer with proper configuration."""
    config = MODEL_CONFIGS[model_name]
    model_args = config["model_args"]
    device_map = model_args.pop("device_map", None)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=model_args.get("cache_dir")
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    device = torch.device("cuda:0") 

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_args,
        device_map={"": device}
    )
    
    # Prepare for k-bit training if using quantization
    if model_args.get("quantization_config"):
        model = prepare_model_for_kbit_training(model)
    
    # Add LoRA configuration
    lora_config = LoraConfig(**config["lora_config"])
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model, tokenizer
    
def prepare_datasets(
    train_dataset: Dataset,
    val_dataset: Dataset,
    tokenizer: Any,
    model_type: str = "t-lite"
) -> tuple[Dataset, Dataset]:
    """Prepare datasets for training."""
    # Токенизируем сразу по ключу "text"
    train_dataset =  train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
        )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
        )

    # print("Sample input_ids:", train_dataset[0]["input_ids"][:20])
    # print("Sample labels:", train_dataset[0]["labels"][:20])

    train_dataset = train_dataset.remove_columns(["text"])
    val_dataset = val_dataset.remove_columns(["text"])
    return train_dataset, val_dataset


def train_model(
    model_name: str,
    train_dataset: Dataset,
    val_dataset: Dataset,
    output_dir: Path,
    run_name: str
) -> None:
    """Train a model using LoRA/QLoRA."""
    logger.info(f"Starting training for {model_name}")
    
    # Load model and tokenizer
    model, tokenizer = get_model_tokenizer(model_name)
    
    # Prepare datasets
    model_type = "t-lite" if "t-lite" in model_name.lower() else "saiga"
    train_dataset, val_dataset = prepare_datasets(
        train_dataset,
        val_dataset,
        tokenizer,
        model_type
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        **MODEL_CONFIGS[model_name]["training_args"]
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )
    
    sample = train_dataset[0]
    print("sample types:")
    for k, v in sample.items():
        print(f"{k}: type={type(v)}, sample={v[:5] if isinstance(v, list) else v}")
    # Train model
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    logger.info("Saving model...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    # Get final metrics
    final_metrics = trainer.evaluate()
    logger.info(f"Training completed. Model saved to {output_dir}")
    
    return final_metrics 