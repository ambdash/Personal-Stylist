import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import wandb
import os
from src.ml.config import MODEL_CONFIGS

def preprocess_data(dataset, tokenizer, max_length=512):
    """Preprocess the data by tokenizing the already formatted input-output pairs"""
    def tokenize(examples):
        model_inputs = tokenizer(
            examples["input_ids"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None
        )
        
        # Tokenize labels
        labels = tokenizer(
            examples["labels"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # Apply tokenization to all examples
    tokenized_dataset = dataset.map(
        tokenize,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    return tokenized_dataset

def finetune_model(model_name, train_dataset, val_dataset, output_dir, wandb_project="llm-finetuning"):
    """
    Fine-tune a model using 4-bit quantization and LoRA.
    
    Args:
        model_name: Name of the base model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        output_dir: Directory to save the model
        wandb_project: W&B project name
    """
    # Load model config
    config = MODEL_CONFIGS[model_name]
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config["name"])
    if not tokenizer.pad_token_id:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        config["name"],
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Create and apply LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    # Preprocess datasets
    max_length = config["model_args"]["max_length"]
    train_dataset = preprocess_data(train_dataset, tokenizer, max_length=max_length)
    if val_dataset:
        val_dataset = preprocess_data(val_dataset, tokenizer, max_length=max_length)

    # Initialize wandb
    wandb.init(
        project=wandb_project,
        name=f"{os.path.basename(config['name'])}-finetune",
        config={
            "model_name": config["name"],
            "lora_config": lora_config.__dict__
        }
    )

    # Create training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch" if val_dataset else "no",
        save_total_limit=3
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8
        )
    )

    # Train
    trainer.train()
    
    # Save the model and adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    wandb.finish() 