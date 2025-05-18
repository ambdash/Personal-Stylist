import os
import sys
import asyncio
import json
import shutil
from typing import List, Dict
import logging
from pathlib import Path
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from dotenv import load_dotenv
import math
import re
import argparse
from datetime import datetime

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
    def __init__(self, batch_size: int = 5, checkpoint_size: int = 10):
        self.client = AsyncOpenAI(
            api_key=os.getenv('OPENAI_API_KEY_ANS_GENERATION'),
            base_url="https://api.proxyapi.ru/openai/v1"   # Using proxy API model
        )
        self.model = "gpt-4o-mini"
        self.batch_size = batch_size
        self.checkpoint_size = checkpoint_size
        self.config = None  # Will be set in load_config

    def load_config(self, config_path: str) -> Dict:
        """Load questions and prompts from config file with a copy of current state"""
        try:
            # Check if input file exists
            input_path = Path(config_path)
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            logger.info(f"Found input file at {input_path}")
            
            # Create a timestamped copy of the input file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            copy_path = input_path.parent / f"{input_path.stem}_copy_{timestamp}.json"
            
            logger.info(f"Attempting to create copy at {copy_path}")
            
            # Copy the current state of the file
            shutil.copy2(input_path, copy_path)
            logger.info(f"Successfully created copy of input file at {copy_path}")
            
            # Verify copy exists
            if not copy_path.exists():
                raise FileNotFoundError(f"Failed to create copy at {copy_path}")
            
            logger.info("Reading from copy file...")
            with open(copy_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Verify config structure
            if "style_questions" not in config:
                raise ValueError("Config file missing 'style_questions' key")
            
            logger.info(f"Successfully loaded {len(config['style_questions'])} questions from config")
            
            # Clean up the copy after loading
            copy_path.unlink()
            logger.info(f"Removed temporary copy file")
            
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
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

    async def generate_dataset(self, input_path: str, output_path: str):
        """Generate complete dataset from questions in config with checkpoints"""
        try:
            # Load config from input file
            logger.info(f"Loading config from {input_path}")
            self.config = self.load_config(input_path)
            
            # Verify output directory exists
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured output directory exists: {output_dir}")
            
            questions = self.config["style_questions"][10699:]
            dataset = []
            num_batches = math.ceil(len(questions) / self.batch_size)
            
            logger.info(f"Starting dataset generation with {len(questions)} questions in {num_batches} batches")

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

            # Final save
            self.save_checkpoint(dataset, output_path)
            logger.info(f"Successfully generated dataset with {len(dataset)} samples")
            return dataset
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise

async def main():
    parser = argparse.ArgumentParser(description='Generate synthetic fashion QA dataset')
    parser.add_argument('--input', type=str, required=True,
                      help='Path to input questions config file')
    parser.add_argument('--output', type=str, required=True,
                      help='Path to output generated dataset file')
    parser.add_argument('--batch-size', type=int, default=5,
                      help='Number of questions to process in each batch')
    parser.add_argument('--checkpoint-size', type=int, default=10,
                      help='Number of questions to process before saving a checkpoint')
    
    args = parser.parse_args()
    
    # Convert paths to absolute paths
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")
    
    try:
        generator = SyntheticDataGenerator(
            batch_size=args.batch_size,
            checkpoint_size=args.checkpoint_size
        )
        await generator.generate_dataset(str(input_path), str(output_path))
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
