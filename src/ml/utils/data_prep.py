import json
from datasets import Dataset

def load_fashion_qa(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Remove repeated instruction in output if needed
    for item in data:
        instr = item['instruction'].strip()
        if item['output'].startswith(instr):
            item['output'] = item['output'][len(instr):].lstrip(" .:-\n")
    return data

def to_hf_dataset(data):
    # HuggingFace expects a list of dicts with 'instruction' and 'output'
    return Dataset.from_list(data) 