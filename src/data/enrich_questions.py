import argparse
import json
import random
from pathlib import Path
from typing import List, Dict
import logging
import asyncio
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QuestionEnricher:
    def __init__(self):
        # Initialize OpenAI client
        self.openai_client = AsyncOpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url="https://api.proxyapi.ru/openai/v1"
        )
        
        # Define replacement categories
        self.categories = {
            "clothing": [
                "платье", "брюки", "юбка", "блузка", "рубашка", "пиджак", "жакет", "свитер",
                "футболка", "джинсы", "шорты", "пальто", "куртка", "костюм", "брючный костюм", "топ"
            ],
            "colors": [
                "белый", "черный", "красный", "синий", "зеленый", "желтый", "розовый", "фиолетовый",
                "оранжевый", "серый", "бежевый", "бордовый", "голубой", "бирюзовый", "коричневый"
            ],
            "aesthetics": [
                "минимализм", "классика", "casual", "богемный", "винтаж", "гранж", "спортивный",
                "деловой", "романтический", "streetwear", "элегантный", "повседневный", "grunge", 
                "preppy", "преппи", "dark academia", "академия", "y2k", "Y2K"
            ],
            "occasions": [
                "офис", "свидание", "вечеринка", "встреча с друзьями", "деловая встреча",
                "свадьба", "отпуск", "прогулка", "шоппинг", "ужин в ресторане"
            ],
            "years": [
                "2024", "2025"
            ]
        }

    async def enrich_question(self, question: str, num_variations: int = 4) -> List[str]:
        """Enrich question with contextual replacements and synonyms"""
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"""Ты - помощник для обогащения вопросов о стиле и моде.
                    Создай {num_variations} разных вариаций вопроса, заменяя слова на синонимы и добавляя контекст.
                    
                    Используй следующие категории для замен:
                    - Одежда: {', '.join(self.categories['clothing'])}
                    - Цвета: {', '.join(self.categories['colors'])}
                    - Стили: {', '.join(self.categories['aesthetics'])}
                    - Поводы: {', '.join(self.categories['occasions'])}
                    - Времена года и годы: {', '.join(self.categories['years'])}
                    
                    Правила:
                    1. Каждая вариация должна использовать разные слова из разных категорий
                    2. Сохраняй основной смысл вопроса
                    3. Каждая вариация должна быть на новой строке
                    4. Начинай каждую вариацию с номера (1., 2., 3. и т.д.)
                    5. Используй естественные сочетания слов
                    6. Добавляй контекст (например, "для офиса", "на свидание", "в стиле минимализм")"""},
                    {"role": "user", "content": f"Обогати этот вопрос {num_variations} раз: {question}"}
                ],
                temperature=0.7,
                max_tokens=1000
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
            logger.warning(f"Question enrichment failed: {e}")
            return [question]

    async def enrich_dataset(self, input_path: str, output_path: str, variations_per_question: int = 3):
        """Enrich entire dataset with variations"""
        # Load questions
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        questions = data["style_questions"][10:]
        enriched_questions = []
        
        # Process each question
        for question in questions:
            # Add original question
            enriched_questions.append(question)
            
            # Generate enriched variations
            variations = await self.enrich_question(question, num_variations=variations_per_question)
            enriched_questions.extend(variations)
        
        # Save enriched dataset
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "style_questions": enriched_questions,
                "system_prompt": data.get("system_prompt", "")
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Generated {len(enriched_questions)} questions (original: {len(questions)})")
        return enriched_questions

async def main():
    parser = argparse.ArgumentParser(description='Enrich fashion questions with variations')
    parser.add_argument('--input', type=str, default='src/data/config/questions.json',
                      help='Path to input questions file')
    parser.add_argument('--output', type=str, default='src/data/config/enriched_questions.json',
                      help='Path to output enriched questions file')
    parser.add_argument('--variations', type=int, default=5,
                      help='Number of variations per question')
    
    args = parser.parse_args()
    
    try:
        enricher = QuestionEnricher()
        await enricher.enrich_dataset(
            args.input,
            args.output,
            variations_per_question=args.variations
        )
        logger.info(f"Successfully generated enriched dataset at {args.output}")
    except Exception as e:
        logger.error(f"Failed to generate enriched dataset: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 