import json
from datasets import Dataset

def load_fashion_qa(json_path, model_name=None):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    system_prompt = """Ты — русскоязычный ассистент по моде и стилю. 
Твоя задача — давать конкретные рекомендации по подбору одежды.
Отвечай четко по следующим пунктам:
1. Основные предметы гардероба
2. Цветовые сочетания
3. Аксессуары
4. Советы по стилизации
5. Где найти похожие вещи

Отвечай кратко и по существу, без лишней информации."""

    formatted_data = []
    for item in data:
        original_instruction = item['instruction'].strip()
        output = item['output'].strip()
        
        # Remove repeated instruction in output if needed
        if output.startswith(original_instruction):
            output = output[len(original_instruction):].lstrip(" .:-\n")

        if model_name and "saiga" in model_name.lower():
            # Format for Saiga models
            formatted_text = {
                "instruction": f"{system_prompt}\n\nВопрос: {original_instruction}",
                "output": f"Ответ: {output}"
            }
        else:
            # Format using standard template
            formatted_text = {
                "instruction": f"{system_prompt}\n\nЗапрос: {original_instruction}",
                "output": output
            }
        
        formatted_data.append(formatted_text)
    
    return formatted_data

def to_hf_dataset(data):
    return Dataset.from_list(data) 