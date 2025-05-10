import os
import sys
import asyncio
import json
from typing import List, Dict
import logging
from pathlib import Path
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from dotenv import load_dotenv
import math
import re

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.utils.retry import async_retry

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url="https://api.proxyapi.ru/openai/v1"   # Using proxy API model
        )
        self.model = "gpt-4o-mini"
        self.batch_size = 5
        self.checkpoint_size = 10  # Save after every 10 questions
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """Load questions and prompts from config file"""
        config_path = Path(__file__).parent / "config" / "questions.json"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            # Fallback to project root config
            config_path = project_root / "data" / "config" / "questions.json"
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise

    @async_retry(retries=3, delay=5)
    async def generate_batch_response(self, prompts: List[str]) -> List[str]:
        """Generate responses for a batch of prompts"""
        joined_prompts = "\n".join([f"{i+1}. {prompt}\nОтвет:" for i, prompt in enumerate(prompts, 1)])

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.config["system_prompt"]},
                    {"role": "user", "content": f"Ответь на вопросы:\n{joined_prompts}"}
                ],
                max_tokens=6000,
                presence_penalty=0.6,
                temperature=0.4,
                top_p=0.9,
                frequency_penalty=0.4 
            )

            answer_text = response.choices[0].message.content.strip()
            return self.split_batch_answers(answer_text, prompts)

        except Exception as e:
            logger.error(f"Error in batch generation: {e}")
            await asyncio.sleep(20)
            return await self.generate_batch_response(prompts)

    def split_batch_answers(self, text: str, prompts: List[str]) -> List[str]:
        """Split text into individual answers and clean them"""
        answers = []
        
        parts = re.split(r'\n\s*\d+\.\s*', '\n' + text)[1:]
        
        for prompt, answer in zip(prompts, parts):
            cleaned = self.clean_answer(answer, prompt)
            answers.append(cleaned)
        
        return answers

    def clean_answer(self, answer: str, question: str) -> str:
        """Remove the question from the answer"""
        answer = answer.strip()
        question = question.rstrip('?')
        if answer.lower().startswith(question.lower()):
            answer = answer[len(question):].strip()
            answer = answer.lstrip('?').strip()
        
        return answer

    def save_checkpoint(self, dataset: List[Dict], output_path: str):
        """Save current progress to file"""
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Checkpoint saved: {len(dataset)} samples")

    async def generate_dataset(self, output_path: str):
        """Generate complete dataset from questions in config with checkpoints"""
        questions = self.config["style_questions"]
        dataset = []
        num_batches = math.ceil(len(questions) / self.batch_size)
        
        logger.info(f"Generating dataset with {len(questions)} questions in {num_batches} batches")

        for i in tqdm_asyncio(range(0, len(questions), self.batch_size)):
            batch = questions[i:i + self.batch_size]
            try:
                max_retries = 3
                for retry in range(max_retries):
                    answers = await self.generate_batch_response(batch)
                    if len(answers) == len(batch):
                        break
                    logger.warning(f"Retry {retry + 1}/{max_retries} for batch {i//self.batch_size + 1}")
                
                if len(answers) != len(batch):
                    raise Exception(f"Failed to get correct number of answers after {max_retries} retries")
                
                for q, a in zip(batch, answers):
                    if not a.strip():
                        raise Exception(f"Empty answer received for question: {q}")
                    dataset.append({
                        "instruction": q,
                        "input": "",
                        "output": a
                    })
                
                if len(dataset) % self.checkpoint_size == 0:
                    self.save_checkpoint(dataset, output_path)
                
                await asyncio.sleep(2)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                # Save checkpoint before retrying
                if dataset:
                    self.save_checkpoint(dataset, output_path)
                i -= self.batch_size
                continue

        # Fin save
        self.save_checkpoint(dataset, output_path)
        logger.info(f"Successfully generated dataset with {len(dataset)} samples")
        return dataset

async def main():
    try:
        generator = SyntheticDataGenerator()
        await generator.generate_dataset("data/fashion_qa_new.json")
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
