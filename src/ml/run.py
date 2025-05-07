import argparse
import sys
from pathlib import Path
import logging
import os
import wandb
from datetime import datetime
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
import json
from datasets import Dataset, load_dataset
import torch.distributed as dist
from accelerate import init_empty_weights
from accelerate.utils import set_seed
from tqdm import tqdm
import time
from bert_score import score

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.ml.data_preparation.prepare_saiga_data import load_fashion_qa, format_saiga_prompt
from src.ml.dataset import split_and_prepare_dataset
from src.ml.finetune import finetune_model
from src.ml.test import evaluate_model, infer_single
from src.ml.config import MODEL_CONFIGS, MAX_LENGTH, CACHE_DIR
from src.ml.utils.fashion_db import FashionKnowledgeBase
from src.ml.utils.fashion_knowledge import FashionKnowledgeManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_output_structure(base_dir: str, model_type: str) -> Path:
    """Create organized output directory structure"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Determine model variant
    if "saiga2" in model_type.lower():
        model_variant = "saiga2_lora"
    elif "mistral" in model_type.lower():
        model_variant = "saiga_mistral"
    else:
        model_variant = "unknown"
    
    # Create path structure: base_dir/model_variant/timestamp
    output_path = Path(base_dir) / "finetuned" / model_variant / timestamp
    
    # Create directories if they don't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for different artifacts
    (output_path / "checkpoints").mkdir(exist_ok=True)
    (output_path / "logs").mkdir(exist_ok=True)
    (output_path / "eval").mkdir(exist_ok=True)
    
    return output_path

def train_saiga(
    model_name: str,
    data_dir: str,
    output_dir: str,
    wandb_project: str = "finetune-saiga",
):
    """Train or finetune Saiga model with LoRA"""
    
    # Initialize distributed training
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(gpu)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
    else:
        rank = 0
        world_size = 1
        gpu = 0
        # Initialize a single process group for non-distributed training
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=1,
            rank=0
        )

    set_seed(42)
    
    # Create organized output structure
    output_path = create_output_structure(output_dir, model_name)
    logger.info(f"Created output directory structure at {output_path}")
    
    try:
        # Load base model with 8-bit quantization
        logger.info(f"Loading base model with existing LoRA adapter: {model_name}")
        model_name = model_name.replace("IlyaGusev/IlyaGusev/", "IlyaGusev/")
        
        # Configure 8-bit quantization with specific settings
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # Load model with updated configuration
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map={"": gpu},  # Explicitly map to current GPU
            torch_dtype=torch.float16,
            cache_dir=CACHE_DIR
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            trust_remote_code=True,
            cache_dir=CACHE_DIR
        )
        
        if not tokenizer.pad_token_id:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load the existing LoRA adapter and prepare for continued training
        logger.info("Loading existing LoRA adapter and preparing for continued training")
        model = PeftModel.from_pretrained(
            model,
            model_name,
            is_trainable=True,
            device_map={"": gpu}  # Explicitly map to current GPU
        )
        
        # Prepare model for training with specific settings
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        
        # Move model to appropriate device
        if torch.cuda.is_available():
            model = model.cuda(gpu)
        
        # Load fashion datasets
        logger.info("Loading fashion datasets...")
        dataset = load_dataset(
            "json",
            data_files={
                "train": str(Path(data_dir) / "train.jsonl"),
                "validation": str(Path(data_dir) / "val.jsonl")
            }
        )
        
        # Get the active adapter configuration
        adapter_name = model.active_adapter or "default"
        lora_cfg = model.peft_config[adapter_name]
        
        # Save training metadata
        metadata = {
            "model_name": model_name,
            "training_type": "continue_training_existing_lora",
            "training_started": datetime.now().isoformat(),
            "dataset_size": {
                "train": len(dataset["train"]),
                "validation": len(dataset["validation"])
            },
            "hyperparameters": {
                "epochs": 3,
                "batch_size": 4,
                "learning_rate": 1e-4,
                "existing_lora_config": {
                    "adapter_name": adapter_name,
                    "r": lora_cfg.r,
                    "lora_alpha": lora_cfg.lora_alpha,
                    "target_modules": list(lora_cfg.target_modules),
                    "lora_dropout": lora_cfg.lora_dropout,
                    "bias": lora_cfg.bias,
                    "task_type": str(lora_cfg.task_type)
                }
            }
        }
        
        # Only save metadata on main process
        if rank == 0:
            with open(output_path / "training_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Initialize wandb only on main process
            model_variant = "saiga2_lora" if "saiga2" in model_name else "saiga_mistral"
            wandb.init(
                project=wandb_project,
                name=f"model_variant_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=metadata,
                dir=str(output_path / "logs"),
                tags=["continue_training", "fashion_domain"]
            )

        # Enable memory optimizations
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

        # Update training arguments for distributed training
        training_args = TrainingArguments(
            output_dir=str(output_path / "checkpoints"),
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            # Remove fp16 settings and use bf16 instead
            fp16=False,
            bf16=True,  # Use bf16 for better stability
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
            report_to="wandb" if rank == 0 else "none",
            gradient_checkpointing=True,
            optim="adamw_torch",
            logging_dir=str(output_path / "logs"),
            eval_steps=100,
            save_total_limit=3,
            metric_for_best_model="eval_loss",
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
            ddp_backend="nccl",
            local_rank=rank,
            # Remove problematic fp16 settings
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            # Add these for better training stability
            warmup_ratio=0.1,
            weight_decay=0.01,
            max_grad_norm=1.0
        )

        # Preprocess dataset to match model input format
        def preprocess_function(examples):
            # Combine system, user, and bot messages into input_ids and labels
            prompts = [
                f"<s>system\n{system}</s><s>user\n{user}</s><s>bot\n{bot}</s>"
                for system, user, bot in zip(examples["system"], examples["user"], examples["bot"])
            ]
            
            # Convert MAX_LENGTH/2 to integer
            max_length = int(MAX_LENGTH / 2)
            
            tokenized = tokenizer(
                prompts,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors=None
            )
            
            # Set up labels
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized

        # Apply preprocessing
        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=processed_dataset["train"],
            eval_dataset=processed_dataset["validation"],
            data_collator=DataCollatorForSeq2Seq(
                tokenizer,
                model=model,
                label_pad_token_id=-100,
                pad_to_multiple_of=8
            )
        )

        # Train
        logger.info("Continuing training on fashion data...")
        trainer.train()

        # Save final model only on main process
        if rank == 0:
            final_model_path = output_path / "final_model"
            trainer.save_model(str(final_model_path))
            tokenizer.save_pretrained(str(final_model_path))
            
            if processed_dataset["validation"] is not None:
                eval_results = trainer.evaluate()
                with open(output_path / "eval" / "final_evaluation.json", "w") as f:
                    json.dump(eval_results, f, indent=2)

        logger.info(f"Training completed. All artifacts saved to {output_path}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        with open(output_path / "error_log.txt", "w") as f:
            f.write(f"Error during training: {str(e)}\n")
        raise
    finally:
        if rank == 0 and wandb.run is not None:
            wandb.finish()
        # Always destroy the process group
        dist.destroy_process_group()

    return output_path

def main():
    parser = argparse.ArgumentParser(description="LLM Fine-tuning and Evaluation")
    parser.add_argument(
        "--mode",
        choices=["train", "test", "infer", "train_saiga"],
        required=True,
        help="Mode: train, test, infer, or train_saiga"
    )
    parser.add_argument(
        "--model",
        choices=["saiga2_7b_lora", "saiga_mistral_7b"] + list(MODEL_CONFIGS.keys()),
        required=True,
        help="Model to use"
    )
    parser.add_argument(
        "--data_dir",
        default="src/data/processed/saiga",
        help="Directory with processed Saiga datasets (train.jsonl and val.jsonl)"
    )
    parser.add_argument(
        "--output_dir",
        default="src/models/finetuned",
        help="Output directory for the model"
    )
    parser.add_argument(
        "--wandb_project",
        default="llm-finetuning",
        help="Wandb project name"
    )
    parser.add_argument(
        "--input_text",
        help="Input text for inference mode"
    )
    parser.add_argument(
        "--use_finetuned",
        action="store_true",
        help="Whether to use finetuned model (for test/infer modes)"
    )
    parser.add_argument(
        "--finetuned_path",
        help="Path to finetuned model (defaults to output_dir if not specified)"
    )
    
    args = parser.parse_args()

    # Determine finetuned model path if needed
    finetuned_path = None
    if args.use_finetuned:
        finetuned_path = args.finetuned_path or args.output_dir
        if not os.path.exists(os.path.join(finetuned_path, "adapter_config.json")):
            logger.error(f"No finetuned model found at {finetuned_path}")
            sys.exit(1)

    if args.mode == "train_saiga":
        model_name = f"IlyaGusev/{args.model}"
        train_saiga(
            model_name=model_name,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            wandb_project=f"{args.wandb_project}-saiga"
        )
    elif args.mode in ["train", "test"]:
        # Load and prepare dataset
        logger.info(f"Loading and preparing dataset from {args.data_dir}")
        train_dataset, val_dataset, test_dataset = split_and_prepare_dataset(
            args.data_dir,
            train_size=0.8,
            val_size=0.1,
            test_size=0.1
        )

        if args.mode == "train":
            logger.info(f"Starting training with model {args.model}")
            finetune_model(
                args.model,
                train_dataset,
                val_dataset,
                args.output_dir,
                args.wandb_project
            )
        else:
            logger.info(f"Starting evaluation with model {args.model}")
            model_type = "finetuned" if args.use_finetuned else "pretrained"
            logger.info(f"Using {model_type} model")
            if args.use_finetuned:
                logger.info(f"Loading finetuned model from {finetuned_path}")
            
            avg_f1 = evaluate_model(
                args.model,
                test_dataset,
                finetuned_path,
                args.wandb_project
            )
            logger.info(f"Evaluation complete. Average BERTScore F1: {avg_f1:.4f}")
    
    elif args.mode == "infer":
        if not args.input_text:
            parser.error("--input_text is required for infer mode")
        
        model_type = "finetuned" if args.use_finetuned else "pretrained"
        logger.info(f"Running inference with {model_type} model {args.model}")
        if args.use_finetuned:
            logger.info(f"Loading finetuned model from {finetuned_path}")
        
        prediction, generation_time = infer_single(
            args.model,
            args.input_text,
            finetuned_path
        )
        logger.info(f"Generated response in {generation_time:.2f} seconds")
        logger.info(f"Response: {prediction}")

        # After generating the prediction, validate and store it
        if "outfit" in prediction.lower():
            try:
                items = parse_outfit_from_prediction(prediction)
                
                # Validate outfit compatibility
                compatibility_issues = []
                for i in range(len(items)):
                    for j in range(i+1, len(items)):
                        compatible_items = FashionKnowledgeManager.find_compatible_items(items[i])
                        if items[j] not in [item["item_name"] for item in compatible_items]:
                            compatibility_issues.append(f"{items[i]} might not work well with {items[j]}")
                
                if compatibility_issues:
                    logger.warning(f"Compatibility issues found: {', '.join(compatibility_issues)}")
                
                # Store in Neo4j
                FashionKnowledgeBase.store_generated_outfit(items, style, occasion)
                
            except Exception as e:
                logger.warning(f"Failed to store outfit in Neo4j: {str(e)}")

    elif args.mode == "test":
        # Load validation dataset for testing
        logger.info(f"Loading validation dataset from {args.data_dir}")
        val_dataset = load_dataset(
            "json",
            data_files={
                "validation": str(Path(args.data_dir) / "val.jsonl")
            }
        )["validation"]

        # Load model and tokenizer
        logger.info(f"Loading model {args.model}")
        model_name = args.model
        model_type = "finetuned" if args.use_finetuned else "pretrained"
        
        if args.use_finetuned:
            logger.info(f"Loading finetuned model from {args.output_dir}")
            # Load base model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                cache_dir=CACHE_DIR
            )
            # Load finetuned adapter
            model = PeftModel.from_pretrained(
                model,
                args.output_dir,
                is_trainable=False
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                cache_dir=CACHE_DIR
            )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            trust_remote_code=True,
            cache_dir=CACHE_DIR
        )

        # Initialize wandb
        if args.wandb_project:
            wandb.init(
                project=args.wandb_project,
                name=f"test_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "model_name": model_name,
                    "model_type": model_type,
                    "test_size": len(val_dataset)
                }
            )

        # Generate predictions and calculate metrics
        predictions = []
        references = []
        generation_times = []

        logger.info("Generating predictions...")
        for example in tqdm(val_dataset):
            # Format input using the same format as training
            input_text = f"<s>system\n{example['system']}</s><s>user\n{example['user']}</s><s>bot\n"
            
            # Generate
            start_time = time.time()
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_length=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )
            generation_time = time.time() - start_time
            
            # Process output
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            prediction = prediction.split("bot\n")[-1].strip()
            
            predictions.append(prediction)
            references.append(example["bot"])
            generation_times.append(generation_time)

        # Calculate BERTScore
        logger.info("Calculating BERTScore...")
        P, R, F1 = score(predictions, references, lang="ru", verbose=True)
        
        # Calculate average metrics
        avg_f1 = float(F1.mean())
        avg_generation_time = sum(generation_times) / len(generation_times)
        
        # Save results
        results = {
            "model_name": model_name,
            "model_type": model_type,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "bert_score_f1": avg_f1,
                "average_generation_time": avg_generation_time
            },
            "detailed_results": [
                {
                    "input": example["user"],
                    "reference": ref,
                    "prediction": pred,
                    "generation_time": gen_time
                }
                for example, ref, pred, gen_time in zip(
                    val_dataset, references, predictions, generation_times
                )
            ]
        }
        
        # Save to file
        output_path = Path(args.output_dir).parent / "eval"
        output_path.mkdir(exist_ok=True)
        with open(output_path / "test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Log to wandb
        if wandb.run is not None:
            wandb.log({
                "bert_score_f1": avg_f1,
                "average_generation_time": avg_generation_time
            })
            wandb.finish()
        
        logger.info(f"Test completed. BERTScore F1: {avg_f1:.4f}")
        logger.info(f"Results saved to {output_path / 'test_results.json'}")

if __name__ == "__main__":
    main() 