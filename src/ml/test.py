import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from bert_score import score
import wandb
from tqdm import tqdm
import os
import json
import time
from datetime import datetime
import pandas as pd
from peft import PeftModel, PeftConfig

def load_model_and_tokenizer(model_name, model_path=None, use_4bit=True):
    """Load either a pretrained or finetuned model with proper configuration"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure quantization
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        bnb_config = None

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Load LoRA adapter if model_path is provided
    if model_path and os.path.exists(os.path.join(model_path, "adapter_config.json")):
        model = PeftModel.from_pretrained(base_model, model_path)
        print(f"Loaded LoRA adapter from {model_path}")
    else:
        model = base_model
        print(f"Using pretrained model {model_name}")

    return model, tokenizer

def save_metrics(results, model_name, model_type):
    """Save metrics to files and print them"""
    # Create results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"eval_results_{model_name.replace('/', '_')}_{model_type}_{timestamp}.json")
    
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Save CSV for easy analysis
    df = pd.DataFrame(results["detailed_results"])
    csv_file = results_file.replace(".json", ".csv")
    df.to_csv(csv_file, index=False)
    
    # Print metrics
    print("\n" + "="*50)
    print(f"Evaluation Results for {model_name} ({model_type})")
    print("="*50)
    print(f"Average BERTScore F1: {results['average_f1']:.4f}")
    print(f"Average Generation Time: {results['average_generation_time']:.4f} seconds")
    print(f"Number of examples: {len(results['detailed_results'])}")
    print(f"\nResults saved to:")
    print(f"- JSON: {results_file}")
    print(f"- CSV: {csv_file}")
    print("="*50)
    
    return results_file, csv_file

def evaluate_model(model_name, test_dataset, model_path=None, wandb_project="llm-finetuning"):
    """
    Evaluate a model (pretrained or finetuned) on the test dataset.
    
    Args:
        model_name: Name of the base model
        test_dataset: Dataset to evaluate on
        model_path: Path to finetuned model or name of pretrained model
        wandb_project: Name of wandb project
    """
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, model_path)
    model_type = "finetuned" if model_path else "pretrained"
    
    # Initialize wandb
    run_name = f"{model_name}-{model_type}-eval"
    wandb.init(project=wandb_project, name=run_name)
    
    # Generate predictions and measure time
    predictions = []
    references = []
    generation_times = []
    inputs_list = []
    
    print("\nGenerating predictions...")
    for example in tqdm(test_dataset, desc="Inference"):
        # Prepare input
        inputs = tokenizer(example["input_ids"], return_tensors="pt").to(model.device)
        
        # Generate with timing
        start_time = time.time()
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
        end_time = time.time()
        
        # Process output
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generation_time = end_time - start_time
        
        predictions.append(prediction)
        references.append(example["labels"])
        generation_times.append(generation_time)
        inputs_list.append(example["input_ids"])
    
    print("\nCalculating BERTScore...")
    P, R, F1 = score(predictions, references, lang="ru", verbose=True)
    
    # Convert to numpy for calculations
    f1_scores = F1.numpy()
    avg_f1 = float(f1_scores.mean())
    avg_generation_time = sum(generation_times) / len(generation_times)
    
    # Prepare results
    results = {
        "model_name": model_name,
        "model_type": model_type,
        "adapter_path": model_path,
        "timestamp": datetime.now().isoformat(),
        "average_f1": avg_f1,
        "average_generation_time": avg_generation_time,
        "detailed_results": [
            {
                "input": inp,
                "prediction": pred,
                "reference": ref,
                "f1_score": float(f1),
                "generation_time": gen_time
            }
            for inp, pred, ref, f1, gen_time in zip(
                inputs_list, predictions, references, f1_scores, generation_times
            )
        ]
    }
    
    results_file, csv_file = save_metrics(results, model_name, model_type)
    
    wandb.log({
        "bert_score_f1": avg_f1,
        "average_generation_time": avg_generation_time,
        "results_table": wandb.Table(dataframe=pd.DataFrame(results["detailed_results"]))
    })
    
    # Upload files to wandb artifacts
    artifact = wandb.Artifact(f"evaluation_results_{model_type}", type="evaluation")
    artifact.add_file(results_file)
    artifact.add_file(csv_file)
    wandb.log_artifact(artifact)
    
    wandb.finish()
    return avg_f1

def infer_single(model_name, input_text, model_path=None):
    """
    Run inference on a single input.
    
    Args:
        model_name: Name of the base model
        input_text: Input text to generate from
        model_path: Optional path to finetuned model with LoRA adapter
    """
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, model_path)
    model_type = "finetuned" if model_path else "pretrained"
    
    # Prepare input
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # Generate with timing
    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_length=256,
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )
    end_time = time.time()
    
    # Process output
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generation_time = end_time - start_time
    
    print("\n" + "="*50)
    print(f"Inference Results ({model_type} model)")
    print("="*50)
    print(f"Input: {input_text}")
    print(f"Output: {prediction}")
    print(f"Generation time: {generation_time:.4f} seconds")
    print("="*50)
    
    return prediction, generation_time 