import argparse
import json
import random
from pathlib import Path
from typing import List, Dict
import logging
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import openai
from dotenv import load_dotenv
import asyncio
from openai import AsyncOpenAI
import os
import time

load_dotenv()

# Set up external directories
PROJECT_DATA_DIR = os.getenv('PROJECT_DATA_DIR', '/data/dprudnikova/project')
CACHE_DIR = os.path.join(PROJECT_DATA_DIR, 'cache')
MODELS_DIR = os.path.join(PROJECT_DATA_DIR, 'models')

# Create directories if they don't exist
for dir_path in [CACHE_DIR, MODELS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Set environment variables for various libraries
os.environ['TRANSFORMERS_CACHE'] = os.path.join(CACHE_DIR, 'transformers')
os.environ['HF_HOME'] = os.path.join(CACHE_DIR, 'huggingface')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QuestionAugmenter:
    def __init__(self, config_path: str, batch_size: int = 10):
        self.config_path = Path(config_path)
        self.questions = self._load_questions()
        self.batch_size = batch_size
        
        # Initialize OpenAI client
        self.openai_client = AsyncOpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url="https://api.proxyapi.ru/openai/v1"
        )
        
        # Initialize Parrot paraphraser with custom paths
        model_path = os.path.join(MODELS_DIR, 'rut5-base-paraphraser')
        if not os.path.exists(model_path):
            logger.info("Downloading Parrot model...")
            self.parrot_tokenizer = AutoTokenizer.from_pretrained(
                "cointegrated/rut5-base-paraphraser",
                cache_dir=os.path.join(CACHE_DIR, 'transformers')
            )
            self.parrot_model = AutoModelForSeq2SeqLM.from_pretrained(
                "cointegrated/rut5-base-paraphraser",
                cache_dir=os.path.join(CACHE_DIR, 'transformers')
            )
            # Save model locally
            self.parrot_tokenizer.save_pretrained(model_path)
            self.parrot_model.save_pretrained(model_path)
        else:
            logger.info("Loading Parrot model from local cache...")
            self.parrot_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.parrot_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        
    def _load_questions(self) -> List[str]:
        """Load questions from config file"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config["style_questions"][10:]

    async def openai_paraphrase(self, text: str, num_variations: int = 3) -> List[str]:
        """Generate multiple paraphrased variations using OpenAI in a single call"""
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"""Ты - помощник для перефразирования текста. 
                    Создай {num_variations} разных вариаций вопроса, сохраняя смысл, но используя другие слова и структуру.
                    Каждая вариация должна быть уникальной и использовать разные подходы:
                    1. Добавь контекст (например, "в духе Pinterest", "2025", "вечерний вариант")
                    2. Измени фокус вопроса (например, добавить аксессуары, цветовые сочетания)
                    3. Используй синонимы и другую структуру предложения
                    4. Добавь конкретные детали (например, "для офиса", "на свидание", "в стиле минимализм")
                    
                    Важно: 
                    - Сохраняй основной смысл вопроса
                    - Каждая вариация должна быть на новой строке
                    - Начинай каждую вариацию с номера (1., 2., 3. и т.д.)
                    - Не повторяйся в вариациях"""},
                    {"role": "user", "content": f"Перефразируй этот вопрос {num_variations} раз: {text}"}
                ],
                temperature=0.7,
                max_tokens=5000
            )
            
            # Parse variations from response
            variations_text = response.choices[0].message.content.strip()
            variations = []
            for line in variations_text.split('\n'):
                # Remove numbering and clean up
                cleaned = line.strip()
                if cleaned and any(cleaned.startswith(str(i) + '.') for i in range(1, 10)):
                    cleaned = cleaned[2:].strip()
                if cleaned and cleaned not in variations:
                    variations.append(cleaned)
            
            return variations[:num_variations]
        except Exception as e:
            logger.warning(f"OpenAI paraphrasing failed: {e}")
            return [text]

    def parrot_paraphrase(self, text: str) -> str:
        """Paraphrase text using Parrot"""
        try:
            inputs = self.parrot_tokenizer(text, return_tensors="pt", padding=True)
            outputs = self.parrot_model.generate(
                **inputs,
                max_length=100,
                num_beams=5,
                temperature=0.7
            )
            return self.parrot_tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.warning(f"Parrot paraphrasing failed: {e}")
            return text

    def save_batch(self, questions: List[str], output_path: Path, batch_num: int, system_prompt: str):
        """Update the output file with a new batch of questions"""
        # Create or load existing data
        if output_path.exists():
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                existing_questions = data.get("style_questions", [])
        else:
            data = {
                "style_questions": [],
                "system_prompt": system_prompt,
                "batch_info": {
                    "last_batch": 0,
                    "last_update": "",
                    "total_questions": 0
                }
            }
            existing_questions = []
        
        # Update with new questions
        existing_questions.extend(questions)
        data["style_questions"] = existing_questions
        data["batch_info"].update({
            "last_batch": batch_num,
            "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_questions": len(existing_questions)
        })
        
        # Save updated data
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Updated file with batch {batch_num}, total questions: {len(existing_questions)}")

    async def augment_dataset(
        self, 
        output_path: str, 
        openai_variations: int = 3,
        use_parrot: bool = True,
        add_prefixes: bool = True
    ):
        """Generate augmented dataset with configurable options and incremental saving"""
        # Load system prompt from original config
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            system_prompt = config.get("system_prompt", "")

        current_batch = []
        batch_num = 1
        total_questions = 0
        
        for question in tqdm(self.questions, desc="Augmenting questions"):
            # Add original question
            current_batch.append(question)
            
            # Generate OpenAI variations
            openai_variations_list = await self.openai_paraphrase(question, num_variations=openai_variations)
            current_batch.extend(openai_variations_list)
            
            # Add Parrot paraphrasing if enabled
            if use_parrot:
                parrot_variation = self.parrot_paraphrase(question)
                if parrot_variation not in current_batch:
                    current_batch.append(parrot_variation)
            
            # Add common prefixes if enabled
            if add_prefixes:
                prefixes = ["Подскажи,", "Посоветуй,", "Как", "Что", "Какие"]
                if not any(question.startswith(p) for p in prefixes):
                    prefixed = f"{random.choice(prefixes)} {question.lower()}"
                    if prefixed not in current_batch:
                        current_batch.append(prefixed)
            
            # Save batch if it reaches the batch size
            if len(current_batch) >= self.batch_size:
                self.save_batch(current_batch, Path(output_path), batch_num, system_prompt)
                total_questions += len(current_batch)
                current_batch = []
                batch_num += 1
        
        # Save remaining questions
        if current_batch:
            self.save_batch(current_batch, Path(output_path), batch_num, system_prompt)
            total_questions += len(current_batch)
        
        logger.info(f"Completed processing {total_questions} questions in {batch_num} batches")
        return total_questions

async def main():
    parser = argparse.ArgumentParser(description='Generate augmented fashion questions dataset')
    parser.add_argument('--config', type=str, default='src/data/config/enriched_questions.json',
                      help='Path to input questions config file')
    parser.add_argument('--output', type=str, default='src/data/config/augmented_questions.json',
                      help='Path to output augmented questions file')
    parser.add_argument('--openai-variations', type=int, default=4,
                      help='Number of OpenAI variations per question')
    parser.add_argument('--use-parrot', action='store_true', default=True,
                      help='Whether to use Parrot paraphrasing')
    parser.add_argument('--add-prefixes', action='store_true', default=True,
                      help='Whether to add question prefixes')
    parser.add_argument('--batch-size', type=int, default=10,
                      help='Number of questions to process before saving a batch')
    
    args = parser.parse_args()
    
    try:
        augmenter = QuestionAugmenter(args.config, batch_size=args.batch_size)
        total_questions = await augmenter.augment_dataset(
            args.output,
            openai_variations=args.openai_variations,
            use_parrot=args.use_parrot,
            add_prefixes=args.add_prefixes
        )
        logger.info(f"Successfully generated {total_questions} questions in {args.output}")
    except Exception as e:
        logger.error(f"Failed to generate augmented dataset: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 