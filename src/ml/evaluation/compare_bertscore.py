import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate
import json
from tqdm import tqdm

MODEL_NAME = "Qwen/Qwen1.5-7B-Chat"
FINETUNED_PATH = "models/finetuned"
DATA_PATH = "data/data/fashion_qa.json"

def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Load data
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = [f"Instruction: {item['instruction']}\nResponse:" for item in data]
    references = [item["output"] for item in data]

    # Load models
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    finetuned_model = AutoModelForCausalLM.from_pretrained(FINETUNED_PATH, torch_dtype=torch.float16, device_map="auto")

    # Generate responses
    base_outputs = [generate(base_model, tokenizer, p) for p in tqdm(prompts, desc="Base")]
    finetuned_outputs = [generate(finetuned_model, tokenizer, p) for p in tqdm(prompts, desc="Finetuned")]

    # Evaluate with BERTScore
    bertscore = evaluate.load("bertscore")
    base_score = bertscore.compute(predictions=base_outputs, references=references, lang="ru")
    finetuned_score = bertscore.compute(predictions=finetuned_outputs, references=references, lang="ru")

    print("Base model BERTScore F1:", sum(base_score["f1"]) / len(base_score["f1"]))
    print("Finetuned model BERTScore F1:", sum(finetuned_score["f1"]) / len(finetuned_score["f1"])) 