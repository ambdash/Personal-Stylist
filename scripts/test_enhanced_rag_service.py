#!/usr/bin/env python3.11
"""
Test script for Enhanced RAG Service
Tests the enhanced_rag_service with various questions and outputs results to a text file.
"""

import asyncio
import sys
import os
import logging
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append('src')

from api.services.enhanced_rag_service import EnhancedRagService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedRagTester:
    def __init__(self):
        self.rag_service = EnhancedRagService()
        self.results = []
        
    def get_test_questions(self):
        """Get a comprehensive set of test questions"""
        return [
            # User's specific question
            "Какую летнюю обувь лучше выбрать для офиса в 2025 году?",
            
            # Questions from enriched_questions.json (various categories)
            "Ищу модный и практичный наряд в стиле 90-х",
            "Подбери стильный образ в стиле Y2K",
            "Что надеть на деловую встречу весной?",
            "Как одеться в дождливую погоду, чтобы не потерять стиль?",
            "Хочу выглядеть дорого, но с масс-маркета. Что надеть?",
            "Какой цвет обуви подходит к зеленому платью?",
            "Как одеться на вечеринку в стиле 90-х?",
            "Мне нужно что-то в духе гранжа на осень",
            "Что надеть, если холодно, но хочется выглядеть элегантно?",
            "Как составить капсульный гардероб на лето?",
            "Что сочетать с серебряным топом?",
            "Нужен минималистичный лук для работы",
            "Как одеться в аэропорт: удобно и стильно",
            "Подбери образ для свидания в кафе летом",
            "Как стилизовать базовый белый топ?",
            "Ищу образ для прогулки осенью в парке",
            "Какая обувь лучше всего подойдет к юбке миди?",
            "Как носить кожаную куртку в стиле гранж?",
            "Образ для вечеринки в клубе в стиле Y2K",
            "Стильный аутфит для работы в офисе летом",
            
            # Additional test cases for different scenarios
            "Какие туфли подойдут к черному платью?",
            "Что надеть зимой в офис?",
            "Как одеться на романтическое свидание?",
            "Какую одежду выбрать для спортзала?",
            "Что надеть на выпускной?",
            "Как одеться для похода в театр?",
            "Какой наряд выбрать для пикника?",
            "Что надеть на собеседование?",
            "Как одеться для путешествия?",
            "Какую обувь носить с джинсами?"
        ]
    
    async def test_single_question(self, question: str, question_num: int) -> dict:
        """Test a single question and return results"""
        logger.info(f"Testing question {question_num}: {question}")
        
        start_time = time.time()
        try:
            result = await self.rag_service.enhance_prompt_async(question)
            processing_time = time.time() - start_time
            
            test_result = {
                'question_number': question_num,
                'question': question,
                'status': result.get('status', 'unknown'),
                'enhanced': result.get('enhanced', False),
                'strategy_used': result.get('strategy_used', 'unknown'),
                'item_type': result.get('item_type', ''),
                'item_subtype': result.get('item_subtype', ''),
                'is_styling': result.get('is_styling', False),
                'styling_item': result.get('styling_item', ''),
                'key_nodes_count': len(result.get('key_nodes_found', [])),
                'key_nodes': result.get('key_nodes_found', []),
                'concepts_count': len(result.get('concepts_used', [])),
                'concepts': result.get('concepts_used', []),
                'original_prompt': result.get('original_prompt', ''),
                'enhanced_prompt': result.get('enhanced_prompt', ''),
                'processing_time': processing_time,
                'error': result.get('error', None)
            }
            
            logger.info(f"Question {question_num} completed: {result.get('status')} - Enhanced: {result.get('enhanced')}")
            return test_result
            
        except Exception as e:
            logger.error(f"Error testing question {question_num}: {str(e)}")
            return {
                'question_number': question_num,
                'question': question,
                'status': 'error',
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    async def run_all_tests(self):
        """Run all test questions"""
        questions = self.get_test_questions()
        logger.info(f"Starting tests for {len(questions)} questions")
        
        # Test questions sequentially to avoid overwhelming the system
        for i, question in enumerate(questions, 1):
            result = await self.test_single_question(question, i)
            self.results.append(result)
            
            # Small delay between requests
            await asyncio.sleep(0.5)
        
        logger.info(f"Completed all {len(questions)} tests")
    
    def save_results_to_file(self, filename: str = None):
        """Save test results to a text file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_rag_test_results_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ENHANCED RAG SERVICE TEST RESULTS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Questions Tested: {len(self.results)}\n")
            
            # Summary statistics
            successful = sum(1 for r in self.results if r.get('status') == 'success')
            enhanced = sum(1 for r in self.results if r.get('enhanced', False))
            errors = sum(1 for r in self.results if r.get('status') == 'error')
            
            f.write(f"Successful: {successful}\n")
            f.write(f"Enhanced: {enhanced}\n")
            f.write(f"Errors: {errors}\n")
            f.write("=" * 80 + "\n\n")
            
            # Strategy usage statistics
            strategies = {}
            for result in self.results:
                strategy = result.get('strategy_used', 'unknown')
                strategies[strategy] = strategies.get(strategy, 0) + 1
            
            f.write("STRATEGY USAGE STATISTICS:\n")
            f.write("-" * 40 + "\n")
            for strategy, count in strategies.items():
                f.write(f"{strategy}: {count}\n")
            f.write("\n")
            
            # Detailed results for each question
            for i, result in enumerate(self.results, 1):
                f.write(f"QUESTION {i}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Question: {result['question']}\n")
                f.write(f"Status: {result.get('status', 'unknown')}\n")
                f.write(f"Enhanced: {result.get('enhanced', False)}\n")
                f.write(f"Strategy: {result.get('strategy_used', 'unknown')}\n")
                f.write(f"Processing Time: {result.get('processing_time', 0):.3f}s\n")
                
                if result.get('error'):
                    f.write(f"Error: {result['error']}\n")
                else:
                    f.write(f"Item Type: {result.get('item_type', 'None')}\n")
                    f.write(f"Item Subtype: {result.get('item_subtype', 'None')}\n")
                    f.write(f"Is Styling: {result.get('is_styling', False)}\n")
                    if result.get('styling_item'):
                        f.write(f"Styling Item: {result['styling_item']}\n")
                    
                    f.write(f"Key Nodes Found: {result.get('key_nodes_count', 0)}\n")
                    if result.get('key_nodes'):
                        for node in result['key_nodes']:
                            f.write(f"  - {node.get('type', 'Unknown')}: {node.get('name', 'Unknown')}\n")
                    
                    f.write(f"Concepts Found: {result.get('concepts_count', 0)}\n")
                    if result.get('concepts'):
                        for j, concept in enumerate(result['concepts'][:3], 1):  # Show top 3
                            f.write(f"  {j}. {concept.get('formatted_text', concept.get('name', 'Unknown'))}\n")
                
                f.write("\nORIGINAL PROMPT:\n")
                f.write(result.get('original_prompt', 'N/A'))
                f.write("\n\nENHANCED PROMPT:\n")
                f.write(result.get('enhanced_prompt', 'N/A'))
                f.write("\n\n" + "=" * 80 + "\n\n")
        
        logger.info(f"Results saved to {filename}")
        return filename

async def main():
    """Main test function"""
    logger.info("Starting Enhanced RAG Service Tests")
    
    tester = EnhancedRagTester()
    
    try:
        # Run all tests
        await tester.run_all_tests()
        
        # Save results
        filename = tester.save_results_to_file()
        
        # Print summary
        successful = sum(1 for r in tester.results if r.get('status') == 'success')
        enhanced = sum(1 for r in tester.results if r.get('enhanced', False))
        errors = sum(1 for r in tester.results if r.get('status') == 'error')
        
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total Questions: {len(tester.results)}")
        print(f"Successful: {successful}")
        print(f"Enhanced: {enhanced}")
        print(f"Errors: {errors}")
        print(f"Results saved to: {filename}")
        print(f"{'='*60}")
        
        # Show strategy breakdown
        strategies = {}
        for result in tester.results:
            strategy = result.get('strategy_used', 'unknown')
            strategies[strategy] = strategies.get(strategy, 0) + 1
        
        print("\nStrategy Usage:")
        for strategy, count in strategies.items():
            print(f"  {strategy}: {count}")
        
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 