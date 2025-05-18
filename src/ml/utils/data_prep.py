import json
from datasets import Dataset
from src.ml.config import SYSTEM_PROMPT

def load_fashion_qa(json_path, model_name=None):
    """Load and format fashion QA data from a JSONL file."""
    formatted_data = []
    
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            original_instruction = item['instruction'].strip()
            output = item['output'].strip()
            
            # Remove repeated instruction in output if needed
            if output.startswith(original_instruction):
                output = output[len(original_instruction):].lstrip(" .:-\n")

            # Use consistent format for all models
            formatted_text = {
                "instruction": original_instruction,
                "output": output,
                "system_prompt": SYSTEM_PROMPT
            }
            
            formatted_data.append(formatted_text)
    
    return formatted_data

def to_hf_dataset(data):
    """Convert formatted data to HuggingFace Dataset."""
    return Dataset.from_list(data) 