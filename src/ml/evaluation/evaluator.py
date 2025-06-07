import torch
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datasets import load_dataset, Dataset
from tqdm import tqdm
import numpy as np
from bert_score import score
from rouge_score import rouge_scorer
import time
from datetime import datetime
from dataclasses import asdict
import evaluate
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.ml.config import TestConfig, SYSTEM_PROMPT, MAX_LENGTH
from src.ml.model_loader.loader import get_model_tokenizer
from src.ml.utils.metrics import calculate_metrics

logger = logging.getLogger(__name__)

def format_prompt(instruction: str, system_prompt: str) -> str:
    """Format the prompt with system message and instruction"""
    return f"{system_prompt}\n\nВопрос: {instruction}\n\nОтвет:"

def get_gpu_utilization():
    """Get current GPU utilization"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
    return 0

def evaluate_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_dataset: Dataset,
    config: TestConfig,
    output_dir: Path,
    run_name: Optional[str] = None
) -> Dict[str, float]:
    """Evaluate model on test dataset."""
    logger.info("Starting model evaluation...")
    
    # Generate run name if not provided
    if run_name is None:
        run_name = f"{model.config.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize metrics
    metrics = {
        "rouge": rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True),
        "bertscore": evaluate.load("bertscore")
    }
    
    # Log model configuration
    logger.info(f"Model device: {model.device}")
    logger.info(f"Model dtype: {model.dtype}")
    logger.info(f"Model config: {model.config}")
    logger.info(f"Tokenizer config: {tokenizer.init_kwargs}")
    
    # Initialize metrics storage
    all_metrics = {
        "rouge-l": [],
        "bertscore": []
    }
    
    # Process dataset in batches
    batch_size = config.batch_size
    num_batches = (len(test_dataset) + batch_size - 1) // batch_size
    
    logger.info(f"Processing {len(test_dataset)} examples in {num_batches} batches")
    
    for i in tqdm(range(0, len(test_dataset), batch_size), desc="Evaluating"):
        batch = test_dataset.select(range(i, min(i + batch_size, len(test_dataset))))
        
        # Prepare inputs and references
        batch_inputs = []
        batch_references = []
        
        for item in batch:
            # Format input with system prompt
            input_text = format_prompt(item["instruction"], SYSTEM_PROMPT)
            batch_inputs.append(input_text)
            batch_references.append(item["output"])
        
        tokenizer.padding_side = "left"
        # Tokenize inputs
        inputs = tokenizer(
            batch_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)
        
        # Generate responses
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                num_beams=config.num_beams,
                do_sample=config.do_sample,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode generated texts
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Clean generated texts (remove input prompt)
        cleaned_texts = []
        for gen_text, input_text in zip(generated_texts, batch_inputs):
            # Remove the input prompt from generated text
            if gen_text.startswith(input_text):
                gen_text = gen_text[len(input_text):].strip()
            cleaned_texts.append(gen_text)
        
        # Calculate metrics
        for ref_text, gen_text in zip(batch_references, cleaned_texts):
            # ROUGE-L
            rouge_scores = metrics["rouge"].score(ref_text, gen_text)
            all_metrics["rouge-l"].append(rouge_scores["rougeL"].fmeasure)
            
            # BERTScore
            bertscore = metrics["bertscore"].compute(
                predictions=[gen_text],
                references=[ref_text],
                lang="ru",
                model_type="xlm-roberta-large"
            )
            all_metrics["bertscore"].append(bertscore["f1"][0])
        
        # Log example analysis
        if i == 0:  # Only for first batch
            logger.info("\nExample Analysis:")
            for j, (input_text, gen_text, ref_text) in enumerate(zip(batch_inputs, cleaned_texts, batch_references)):
                logger.info(f"\nExample {j+1}:")
                logger.info(f"Input: {input_text}")
                logger.info(f"Generated: {gen_text}")
                logger.info(f"Reference: {ref_text}")
                logger.info(f"ROUGE-L: {all_metrics['rouge-l'][j]:.3f}")
                logger.info(f"BERTScore: {all_metrics['bertscore'][j]:.3f}")
    
    # Calculate final metrics
    final_metrics = {
        "rouge-l": np.mean(all_metrics["rouge-l"]),
        "bertscore": np.mean(all_metrics["bertscore"])
    }
    
    # Log final metrics
    logger.info("\nFinal Metrics:")
    for metric_name, value in final_metrics.items():
        logger.info(f"{metric_name}: {value:.3f}")
    
    # Save results
    results = {
        "metrics": final_metrics,
        "config": asdict(config),
        "examples": {
            "inputs": batch_inputs[:3],
            "generated": cleaned_texts[:3],
            "references": batch_references[:3]
        }
    }
    
    results_file = output_dir / f"{run_name}_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    return final_metrics 