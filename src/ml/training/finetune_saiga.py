import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from datasets import load_dataset
import logging
from pathlib import Path
import argparse
import wandb
from transformers import Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_and_prepare_model(model_name: str):
    """Create and prepare model with LoRA config"""
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        trust_remote_code=True
    )
    
    if not tokenizer.pad_token_id:
        tokenizer.pad_token = tokenizer.eos_token
    
    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    return model, tokenizer

def prepare_dataset(tokenizer, data_path: str, max_length: int = 3584):
    """Prepare dataset for training"""
    
    def generate_prompt(example):
        return f"""<s>system
{example['system']}</s><s>user
{example['user']}</s><s>bot
{example['bot']}</s>"""

    def tokenize(example):
        prompt = generate_prompt(example)
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None
        )
        
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id 
            and len(result["input_ids"]) < max_length
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        
        result["labels"] = result["input_ids"].copy()
        return result

    # Load and process dataset
    dataset = load_dataset(
        "json",
        data_files={
            "train": f"{data_path}/train.jsonl",
            "validation": f"{data_path}/val.jsonl"
        }
    )
    
    processed_dataset = dataset.map(
        tokenize,
        remove_columns=dataset["train"].column_names
    )
    
    return processed_dataset

def train(
    model_name: str = "IlyaGusev/saiga2_7b_lora",
    data_dir: str = "src/data/processed",
    output_dir: str = "src/models/finetuned",
    epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    max_length: int = 3584,
):
    """Main training function"""
    
    # Initialize wandb
    wandb.init(
        project=wandb_project,
        config={
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_length": max_length,
        }
    )
    
    logger.info(f"Starting training with model {model_name}")
    
    # Create and prepare model
    model, tokenizer = create_and_prepare_model(model_name)
    
    # Prepare dataset
    dataset = prepare_dataset(tokenizer, data_dir, max_length)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        report_to="wandb",
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="Finetune Saiga models")
    parser.add_argument(
        "--model-name",
        default="IlyaGusev/saiga2_7b_lora",
        help="Model name to finetune"
    )
    parser.add_argument(
        "--data-dir",
        default="src/data/processed",
        help="Directory with train.jsonl and val.jsonl"
    )
    parser.add_argument(
        "--output-dir",
        default="src/models/finetuned",
        help="Output directory for finetuned model"
    )
    parser.add_argument(
        "--wandb-project",
        default="fashion-stylist",
        help="W&B project name"
    )
    
    args = parser.parse_args()
    train(
        model_name=args.model_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project
    )

if __name__ == "__main__":
    main() 