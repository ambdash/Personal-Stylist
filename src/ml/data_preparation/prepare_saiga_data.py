import json
from pathlib import Path
from typing import List, Dict
import logging
from datasets import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_fashion_qa(json_path: str) -> List[Dict]:
    """Load fashion QA dataset"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def format_saiga_prompt(instruction: str, output: str) -> Dict:
    """Format data for Saiga models"""
    system_prompt = """Ты — профессиональный стилист. Отвечай на каждый вопрос, начиная с номера вопроса. Используй структурированный формат для каждого ответа:
    - Предметы гардероба
    - Цветовые сочетания
    - Аксессуары
    - Советы по стилизации
    - Где можно найти подобные вещи
    Пиши на русском языке, стиль — лаконичный, минималистичный. Начинай сразу с сути."""

    return {
        "system": system_prompt,
        "user": instruction,
        "bot": output
    }

def prepare_saiga_dataset(data: List[Dict], output_path: str):
    """Convert data to Saiga format and save"""
    formatted_data = []
    
    for item in data:
        formatted = format_saiga_prompt(
            instruction=item['instruction'],
            output=item['output']
        )
        formatted_data.append(formatted)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in formatted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    input_path = "src/data/data/fashion_qa.json"
    output_dir = Path("src/data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")
    data = load_fashion_qa(input_path)
    
    # Split into train/val (90/10)
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    # Prepare and save datasets
    logger.info("Preparing training data...")
    prepare_saiga_dataset(train_data, output_dir / "train.jsonl")
    
    logger.info("Preparing validation data...")
    prepare_saiga_dataset(val_data, output_dir / "val.jsonl")
    
    logger.info(f"Datasets saved to {output_dir}")
    logger.info(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")

if __name__ == "__main__":
    main() 