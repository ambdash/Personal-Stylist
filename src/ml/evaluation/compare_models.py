from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import evaluate
import json
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(
        self,
        base_model_path: str = "microsoft/phi-2",
        finetuned_model_path: str = "models/finetuned"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.finetuned_model = AutoModelForCausalLM.from_pretrained(
            finetuned_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.bert_score = evaluate.load("bertscore")

    def generate_response(self, model: AutoModelForCausalLM, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_length=512,
            temperature=0.7,
            num_return_sequences=1
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def evaluate_models(self, test_questions: List[str]) -> Dict:
        results = []
        
        for question in tqdm(test_questions):
            base_response = self.generate_response(self.base_model, question)
            finetuned_response = self.generate_response(self.finetuned_model, question)
            
            # Calculate BERT Score
            bert_scores = self.bert_score.compute(
                predictions=[finetuned_response],
                references=[base_response],
                lang="ru"
            )
            
            results.append({
                "question": question,
                "base_response": base_response,
                "finetuned_response": finetuned_response,
                "bert_score_f1": bert_scores["f1"][0]
            })
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv("evaluation_results.csv", index=False)
        
        # Calculate average metrics
        avg_bert_score = df["bert_score_f1"].mean()
        
        return {
            "average_bert_score": avg_bert_score,
            "detailed_results": results
        }

if __name__ == "__main__":
    # Load test questions
    with open("data/synthetic/fashion_qa.json", "r") as f:
        test_data = json.load(f)
    
    test_questions = [item["instruction"] for item in test_data]
    
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_models(test_questions)
    
    logger.info(f"Average BERT Score: {results['average_bert_score']}") 