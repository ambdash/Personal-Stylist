import argparse
import json
import random
from pathlib import Path
from typing import List, Dict
import logging
import torch
import numpy as np
from tqdm import tqdm
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import openai
from dotenv import load_dotenv
import asyncio
from openai import AsyncOpenAI
import os
import nltk

load_dotenv()

# Set up external directories
PROJECT_DATA_DIR = os.getenv('PROJECT_DATA_DIR', '/data/dprudnikova/project')
CACHE_DIR = os.path.join(PROJECT_DATA_DIR, 'cache')
MODELS_DIR = os.path.join(PROJECT_DATA_DIR, 'models')
NLTK_DATA = os.path.join(PROJECT_DATA_DIR, 'nltk_data')

# Create directories if they don't exist
for dir_path in [CACHE_DIR, MODELS_DIR, NLTK_DATA]:
    os.makedirs(dir_path, exist_ok=True)

# Set environment variables for various libraries
os.environ['TRANSFORMERS_CACHE'] = os.path.join(CACHE_DIR, 'transformers')
os.environ['HF_HOME'] = os.path.join(CACHE_DIR, 'huggingface')
os.environ['NLTK_DATA'] = NLTK_DATA

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QuestionAugmenter:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.questions = self._load_questions()
        
        # Initialize OpenAI client
        self.openai_client = AsyncOpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url="https://api.proxyapi.ru/openai/v1"
        )
        
        # Download required NLTK data
        try:
            nltk.download('wordnet', download_dir=NLTK_DATA)
            nltk.download('omw-1.4', download_dir=NLTK_DATA)
        except Exception as e:
            logger.warning(f"Failed to download NLTK data: {e}")
        
        # Initialize NLPAug augmenters with custom paths
        self.wordnet_aug = naw.SynonymAug(
            aug_src='wordnet',
            lang='rus',
            cache_dir=os.path.join(CACHE_DIR, 'nlpaug')
        )
        
        self.contextual_aug = naw.ContextualWordEmbsAug(
            model_path='cointegrated/rubert-tiny2',
            action="substitute",
            cache_dir=os.path.join(CACHE_DIR, 'nlpaug')
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
        return config["style_questions"][:10]

    async def openai_paraphrase(self, text: str) -> str:
        """Paraphrase text using OpenAI"""
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Ты - помощник для перефразирования текста. Перепиши текст, сохраняя смысл, но используя другие слова и структуру. Хорошо, если аугментации будут не просто перефразами, а с немного разным фокусом (например, добавить «в духе Pinterest», «2025», «вечерний вариант», «добавить аксессуары»)."},
                    {"role": "user", "content": f"Перефразируй этот вопрос: {text}"}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"OpenAI paraphrasing failed: {e}")
            return text

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

    def generate_variations(self, question: str) -> List[str]:
        """Generate variations of the question using multiple techniques"""
        variations = []
        
        # 1. WordNet synonym replacement
        try:
            wordnet_variation = self.wordnet_aug.augment(question)[0]
            variations.append(wordnet_variation)
        except Exception as e:
            logger.warning(f"WordNet augmentation failed: {e}")

        # 2. Contextual word replacement
        try:
            contextual_variation = self.contextual_aug.augment(question)[0]
            variations.append(contextual_variation)
        except Exception as e:
            logger.warning(f"Contextual augmentation failed: {e}")

        # 4. Parrot paraphrasing
        try:
            parrot_variation = self.parrot_paraphrase(question)
            variations.append(parrot_variation)
        except Exception as e:
            logger.warning(f"Parrot paraphrasing failed: {e}")

        return variations

    async def augment_dataset(self, output_path: str, augmentations_per_question: int = 3):
        """Generate augmented dataset"""
        augmented_questions = []
        
        for question in tqdm(self.questions, desc="Augmenting questions"):
            # Add original question
            augmented_questions.append(question)
            
            # Generate variations
            variations = self.generate_variations(question)
            
            # Add OpenAI paraphrasing
            openai_variation = await self.openai_paraphrase(question)
            variations.append(openai_variation)
            
            # Add common prefixes
            prefixes = ["Подскажи,", "Посоветуй,", "Как", "Что", "Какие"]
            if not any(question.startswith(p) for p in prefixes):
                variations.append(f"{random.choice(prefixes)} {question.lower()}")
            
            # Add random variations
            augmented_questions.extend(variations[:augmentations_per_question])

        # Save augmented dataset
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "style_questions": augmented_questions,
                "system_prompt": "Ты — профессиональный стилист. Отвечай на каждый вопрос, начиная с номера вопроса. Используй структурированный формат для каждого ответа:\n\n- Предметы гардероба\n- Цветовые сочетания\n- Аксессуары\n- Советы по стилизации\n- Где можно найти подобные вещи\n\nПиши на русском языке, стиль — лаконичный, минималистичный. Начинай сразу с сути."
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"Generated {len(augmented_questions)} questions (original: {len(self.questions)})")
        return augmented_questions

async def main():
    parser = argparse.ArgumentParser(description='Generate augmented fashion questions dataset')
    parser.add_argument('--config', type=str, default='src/data/config/questions.json',
                      help='Path to input questions config file')
    parser.add_argument('--output', type=str, default='src/data/config/augmented_questions.json',
                      help='Path to output augmented questions file')
    parser.add_argument('--augmentations', type=int, default=5,
                      help='Number of augmentations per question')
    
    args = parser.parse_args()
    
    try:
        augmenter = QuestionAugmenter(args.config)
        await augmenter.augment_dataset(args.output, args.augmentations)
        logger.info(f"Successfully generated augmented dataset at {args.output}")
    except Exception as e:
        logger.error(f"Failed to generate augmented dataset: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 