import argparse
import sys
from pathlib import Path
import logging
import os
import wandb
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
from datasets import load_dataset
from peft import PeftModel

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.ml.config import MODEL_CONFIGS, SYSTEM_PROMPT, TestConfig
from src.ml.training.trainer import train_model, get_model_tokenizer
from src.ml.evaluation.evaluator import evaluate_model

def get_model_specific_data_path(data_dir: str, model_name: str, split: str) -> str:
    """Get the correct data path based on model type."""
    base_path = Path(data_dir)
    
    # First try with t_lite_ prefix
    t_lite_path = base_path / f"t_lite_{split}.jsonl"
    if t_lite_path.exists():
        return str(t_lite_path)
    
    # If not found, try without prefix
    plain_path = base_path / f"{split}.jsonl"
    if plain_path.exists():
        return str(plain_path)
    
    # If still not found, try with model-specific prefix
    model_type = "t_lite" if "t-lite" in model_name.lower() else "saiga"
    model_path = base_path / f"{model_type}_{split}.jsonl"
    if model_path.exists():
        return str(model_path)
    
    raise FileNotFoundError(
        f"Could not find test file in any of these locations:\n"
        f"1. {t_lite_path}\n"
        f"2. {plain_path}\n"
        f"3. {model_path}"
    )

def load_model_with_adapter(model_name: str, adapter_path: str):
    """Load base model and LoRA adapter."""
    logger.info(f"Loading base model {model_name}...")
    model, tokenizer = get_model_tokenizer(model_name)
    
    logger.info(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Run model training or evaluation")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"], help="Mode: train or test")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--run_name", type=str, help="Name for this run (used for wandb and output files)")
    parser.add_argument("--wandb_project", type=str, default="fashion-qa", help="W&B project name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--adapter_path", type=str, help="Path to LoRA adapter for evaluation")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb if not disabled
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config={
                "model": args.model,
                "mode": args.mode,
                "data_dir": args.data_dir,
                "output_dir": str(output_dir),
                "adapter_path": args.adapter_path
            }
        )
    
    try:
        if args.mode == "train":
            # Load datasets with model-specific paths
            logger.info("Loading training and validation datasets...")
            train_dataset = load_dataset(
                "json",
                data_files=get_model_specific_data_path(args.data_dir, args.model, "train"),
                split="train"
            )
            val_dataset = load_dataset(
                "json",
                data_files=get_model_specific_data_path(args.data_dir, args.model, "val"),
                split="train"
            )
            
            # Train model
            metrics = train_model(
                model_name=args.model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                output_dir=output_dir,
                run_name=args.run_name
            )
            
            # Log training metrics if wandb is enabled
            if not args.no_wandb:
                wandb.log(metrics)
            
        elif args.mode == "test":
            # Load test dataset with model-specific path
            logger.info("Loading test dataset...")
            test_dataset = load_dataset(
                "json",
                data_files=get_model_specific_data_path(args.data_dir, args.model, "test"),
                split="train"
            )
            
            # Load model with adapter if provided
            if args.adapter_path:
                model, tokenizer = load_model_with_adapter(args.model, args.adapter_path)
            else:
                model, tokenizer = get_model_tokenizer(args.model)
            
            # Create test config
            test_config = TestConfig(
                model_name=args.model,
                batch_size=2,
                max_length=2048,
                max_new_tokens=512,
                num_beams=2,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.2,
                metrics=["rouge", "bertscore"],
                test_size=0.2,
                system_prompt=SYSTEM_PROMPT
            )
            
            # Evaluate model
            logger.info("Evaluating model...")
            metrics = evaluate_model(
                model=model,
                tokenizer=tokenizer,
                test_dataset=test_dataset,
                config=test_config,
                output_dir=output_dir,
                run_name=args.run_name
            )
            
            # Log evaluation metrics if wandb is enabled
            if not args.no_wandb:
                wandb.log(metrics)
        
        logger.info("Pipeline completed successfully")
        
    finally:
        # Always finish wandb run if it was initialized
        if not args.no_wandb:
            wandb.finish()

if __name__ == "__main__":
    main() 