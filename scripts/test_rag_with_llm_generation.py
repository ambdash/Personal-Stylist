#!/usr/bin/env python3.11
"""
Test script for Enhanced RAG Service with actual LLM generation
Tests the enhanced_rag_service with questions from test.jsonl and generates actual answers using LLM.
"""

import asyncio
import sys
import os
import json
import logging
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm

# Add src to path
sys.path.append('src')

from api.services.enhanced_rag_service import EnhancedRagService

# Import metrics calculation modules
try:
    from bert_score import score as bert_score
    from rouge_score import rouge_scorer
    METRICS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Metrics libraries not available: {e}")
    METRICS_AVAILABLE = False

# Import LLM generation (adjust based on your setup)
try:
    import requests
    import aiohttp
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RagLlmTester:
    def __init__(self, jsonl_path: str, llm_endpoint: str = "http://localhost:8000/generate"):
        self.jsonl_path = jsonl_path
        self.llm_endpoint = llm_endpoint
        self.rag_service = EnhancedRagService()
        self.results = []
        
        # Initialize metrics calculators if available
        if METRICS_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        else:
            self.rouge_scorer = None
            logger.warning("Metrics calculation will be skipped due to missing dependencies")
    
    def load_test_data(self) -> List[Dict]:
        """Load test data from JSONL file"""
        logger.info(f"Loading test data from {self.jsonl_path}")
        test_data = []
        
        try:
            with open(self.jsonl_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        if 'instruction' in data and 'output' in data:
                            test_data.append({
                                'line_number': line_num,
                                'instruction': data['instruction'],
                                'expected_output': data['output'],
                                'system_prompt': data.get('system_prompt', '')
                            })
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                        continue
        except FileNotFoundError:
            logger.error(f"Test file not found: {self.jsonl_path}")
            return []
        
        logger.info(f"Loaded {len(test_data)} test examples")
        return test_data
    
    async def generate_with_llm(self, prompt: str, system_prompt: str = "") -> str:
        """Generate answer using LLM API"""
        if not LLM_AVAILABLE:
            logger.warning("LLM generation not available, returning enhanced prompt")
            return prompt
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "max_tokens": 512,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
                
                async with session.post(self.llm_endpoint, json=payload, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get('generated_text', prompt)
                    else:
                        logger.warning(f"LLM API returned status {response.status}")
                        return prompt
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}")
            return prompt
    
    async def test_single_question(self, test_item: Dict, question_num: int) -> Dict:
        """Test a single question with RAG enhancement and LLM generation"""
        instruction = test_item['instruction']
        expected_output = test_item['expected_output']
        system_prompt = test_item.get('system_prompt', '')
        
        logger.info(f"Testing question {question_num}: {instruction[:50]}...")
        
        start_time = time.time()
        try:
            # Step 1: RAG enhancement
            rag_start = time.time()
            rag_result = await self.rag_service.enhance_prompt_async(instruction)
            rag_time = time.time() - rag_start
            
            # Step 2: Get the final prompt (enhanced or original)
            final_prompt = rag_result.get('enhanced_prompt', instruction)
            
            # Step 3: Generate answer with LLM
            llm_start = time.time()
            generated_answer = await self.generate_with_llm(final_prompt, system_prompt)
            llm_time = time.time() - llm_start
            
            total_time = time.time() - start_time
            
            # Step 4: Calculate metrics if available
            metrics = {}
            if METRICS_AVAILABLE and self.rouge_scorer:
                # ROUGE-L score
                rouge_scores = self.rouge_scorer.score(expected_output, generated_answer)
                metrics['rouge_l'] = rouge_scores['rougeL'].fmeasure
                metrics['rouge_1'] = rouge_scores['rouge1'].fmeasure
                metrics['rouge_2'] = rouge_scores['rouge2'].fmeasure
                
                # BERTScore
                try:
                    P, R, F1 = bert_score([generated_answer], [expected_output], lang="ru", verbose=False)
                    metrics['bertscore_f1'] = F1.item()
                    metrics['bertscore_precision'] = P.item()
                    metrics['bertscore_recall'] = R.item()
                except Exception as e:
                    logger.warning(f"BERTScore calculation failed: {e}")
                    metrics['bertscore_f1'] = 0.0
                    metrics['bertscore_precision'] = 0.0
                    metrics['bertscore_recall'] = 0.0
            
            test_result = {
                'question_number': question_num,
                'line_number': test_item['line_number'],
                'instruction': instruction,
                'expected_output': expected_output,
                'generated_answer': generated_answer,
                'rag_enhanced': rag_result.get('enhanced', False),
                'rag_status': rag_result.get('status', 'unknown'),
                'strategy_used': rag_result.get('strategy_used', 'unknown'),
                'item_type': rag_result.get('item_type', ''),
                'item_subtype': rag_result.get('item_subtype', ''),
                'is_styling': rag_result.get('is_styling', False),
                'styling_item': rag_result.get('styling_item', ''),
                'key_nodes_count': len(rag_result.get('key_nodes_found', [])),
                'key_nodes': rag_result.get('key_nodes_found', []),
                'concepts_count': len(rag_result.get('concepts_used', [])),
                'concepts': rag_result.get('concepts_used', []),
                'original_prompt': rag_result.get('original_prompt', ''),
                'enhanced_prompt': rag_result.get('enhanced_prompt', ''),
                'rag_processing_time': rag_time,
                'llm_processing_time': llm_time,
                'total_processing_time': total_time,
                'metrics': metrics,
                'error': rag_result.get('error', None)
            }
            
            logger.info(f"Question {question_num} completed: Enhanced={rag_result.get('enhanced', False)}, "
                       f"RAG: {rag_time:.3f}s, LLM: {llm_time:.3f}s")
            return test_result
            
        except Exception as e:
            logger.error(f"Error testing question {question_num}: {str(e)}")
            return {
                'question_number': question_num,
                'line_number': test_item['line_number'],
                'instruction': instruction,
                'expected_output': expected_output,
                'rag_enhanced': False,
                'rag_status': 'error',
                'error': str(e),
                'total_processing_time': time.time() - start_time,
                'metrics': {}
            }
    
    async def run_all_tests(self, max_questions: Optional[int] = None):
        """Run tests on all questions from JSONL file"""
        test_data = self.load_test_data()
        if not test_data:
            logger.error("No test data loaded")
            return
        
        # Limit number of questions if specified
        if max_questions:
            test_data = test_data[:max_questions]
            logger.info(f"Limited to first {max_questions} questions")
        
        logger.info(f"Starting tests for {len(test_data)} questions")
        
        # Test questions with progress bar
        for i, test_item in enumerate(tqdm(test_data, desc="Testing questions"), 1):
            result = await self.test_single_question(test_item, i)
            self.results.append(result)
            
            # Small delay between requests to avoid overwhelming the system
            await asyncio.sleep(0.2)
        
        logger.info(f"Completed all {len(test_data)} tests")
    
    def calculate_summary_metrics(self) -> Dict[str, Any]:
        """Calculate summary metrics across all test results"""
        if not self.results:
            return {}
        
        # Basic statistics
        total_questions = len(self.results)
        successful_tests = sum(1 for r in self.results if r.get('rag_status') == 'success')
        enhanced_questions = sum(1 for r in self.results if r.get('rag_enhanced', False))
        error_count = sum(1 for r in self.results if r.get('error'))
        
        # Coverage metrics
        coverage_rate = enhanced_questions / total_questions if total_questions > 0 else 0
        success_rate = successful_tests / total_questions if total_questions > 0 else 0
        
        # Processing time statistics
        rag_times = [r.get('rag_processing_time', 0) for r in self.results if r.get('rag_processing_time')]
        llm_times = [r.get('llm_processing_time', 0) for r in self.results if r.get('llm_processing_time')]
        total_times = [r.get('total_processing_time', 0) for r in self.results if r.get('total_processing_time')]
        
        time_stats = {}
        if rag_times:
            time_stats['avg_rag_time'] = np.mean(rag_times)
            time_stats['median_rag_time'] = np.median(rag_times)
        if llm_times:
            time_stats['avg_llm_time'] = np.mean(llm_times)
            time_stats['median_llm_time'] = np.median(llm_times)
        if total_times:
            time_stats['avg_total_time'] = np.mean(total_times)
            time_stats['median_total_time'] = np.median(total_times)
        
        # Strategy usage
        strategies = {}
        for result in self.results:
            strategy = result.get('strategy_used', 'unknown')
            strategies[strategy] = strategies.get(strategy, 0) + 1
        
        # Quality metrics (if available)
        quality_metrics = {}
        if METRICS_AVAILABLE:
            rouge_l_scores = [r.get('metrics', {}).get('rouge_l', 0) for r in self.results if r.get('metrics')]
            rouge_1_scores = [r.get('metrics', {}).get('rouge_1', 0) for r in self.results if r.get('metrics')]
            rouge_2_scores = [r.get('metrics', {}).get('rouge_2', 0) for r in self.results if r.get('metrics')]
            bert_f1_scores = [r.get('metrics', {}).get('bertscore_f1', 0) for r in self.results if r.get('metrics')]
            
            if rouge_l_scores:
                quality_metrics['avg_rouge_l'] = np.mean(rouge_l_scores)
                quality_metrics['median_rouge_l'] = np.median(rouge_l_scores)
                quality_metrics['std_rouge_l'] = np.std(rouge_l_scores)
            
            if rouge_1_scores:
                quality_metrics['avg_rouge_1'] = np.mean(rouge_1_scores)
            
            if rouge_2_scores:
                quality_metrics['avg_rouge_2'] = np.mean(rouge_2_scores)
            
            if bert_f1_scores:
                quality_metrics['avg_bertscore_f1'] = np.mean(bert_f1_scores)
                quality_metrics['median_bertscore_f1'] = np.median(bert_f1_scores)
                quality_metrics['std_bertscore_f1'] = np.std(bert_f1_scores)
        
        return {
            'total_questions': total_questions,
            'successful_tests': successful_tests,
            'enhanced_questions': enhanced_questions,
            'error_count': error_count,
            'coverage_rate': coverage_rate,
            'success_rate': success_rate,
            'time_statistics': time_stats,
            'strategy_usage': strategies,
            'quality_metrics': quality_metrics
        }
    
    def save_results_to_file(self, filename: str = None):
        """Save comprehensive test results to a file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_llm_test_results_{timestamp}.txt"
        
        summary_metrics = self.calculate_summary_metrics()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("RAG + LLM SYSTEM TEST RESULTS\n")
            f.write("=" * 100 + "\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test File: {self.jsonl_path}\n")
            f.write(f"LLM Endpoint: {self.llm_endpoint}\n")
            f.write(f"Total Questions Tested: {summary_metrics.get('total_questions', 0)}\n")
            f.write("\n")
            
            # Summary Statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 50 + "\n")
            f.write(f"Successful Tests: {summary_metrics.get('successful_tests', 0)}\n")
            f.write(f"Enhanced Questions: {summary_metrics.get('enhanced_questions', 0)}\n")
            f.write(f"Errors: {summary_metrics.get('error_count', 0)}\n")
            f.write(f"Coverage Rate: {summary_metrics.get('coverage_rate', 0):.2%}\n")
            f.write(f"Success Rate: {summary_metrics.get('success_rate', 0):.2%}\n")
            f.write("\n")
            
            # Time Statistics
            time_stats = summary_metrics.get('time_statistics', {})
            if time_stats:
                f.write("PROCESSING TIME STATISTICS\n")
                f.write("-" * 50 + "\n")
                if 'avg_rag_time' in time_stats:
                    f.write(f"Average RAG Time: {time_stats['avg_rag_time']:.3f}s\n")
                    f.write(f"Median RAG Time: {time_stats['median_rag_time']:.3f}s\n")
                if 'avg_llm_time' in time_stats:
                    f.write(f"Average LLM Time: {time_stats['avg_llm_time']:.3f}s\n")
                    f.write(f"Median LLM Time: {time_stats['median_llm_time']:.3f}s\n")
                if 'avg_total_time' in time_stats:
                    f.write(f"Average Total Time: {time_stats['avg_total_time']:.3f}s\n")
                    f.write(f"Median Total Time: {time_stats['median_total_time']:.3f}s\n")
                f.write("\n")
            
            # Strategy Usage
            f.write("STRATEGY USAGE STATISTICS\n")
            f.write("-" * 50 + "\n")
            for strategy, count in summary_metrics.get('strategy_usage', {}).items():
                percentage = (count / summary_metrics.get('total_questions', 1)) * 100
                f.write(f"{strategy}: {count} ({percentage:.1f}%)\n")
            f.write("\n")
            
            # Quality Metrics
            if summary_metrics.get('quality_metrics'):
                f.write("QUALITY METRICS\n")
                f.write("-" * 50 + "\n")
                quality = summary_metrics['quality_metrics']
                if 'avg_rouge_l' in quality:
                    f.write(f"Average ROUGE-L: {quality['avg_rouge_l']:.3f}\n")
                    f.write(f"Median ROUGE-L: {quality['median_rouge_l']:.3f}\n")
                    f.write(f"Std ROUGE-L: {quality['std_rouge_l']:.3f}\n")
                if 'avg_rouge_1' in quality:
                    f.write(f"Average ROUGE-1: {quality['avg_rouge_1']:.3f}\n")
                if 'avg_rouge_2' in quality:
                    f.write(f"Average ROUGE-2: {quality['avg_rouge_2']:.3f}\n")
                if 'avg_bertscore_f1' in quality:
                    f.write(f"Average BERTScore F1: {quality['avg_bertscore_f1']:.3f}\n")
                    f.write(f"Median BERTScore F1: {quality['median_bertscore_f1']:.3f}\n")
                    f.write(f"Std BERTScore F1: {quality['std_bertscore_f1']:.3f}\n")
                f.write("\n")
            
            # Sample Results (first 5)
            f.write("SAMPLE RESULTS (First 5 Questions)\n")
            f.write("=" * 100 + "\n")
            
            for i, result in enumerate(self.results[:5], 1):
                f.write(f"\nQUESTION {i} (Line {result.get('line_number', 'N/A')})\n")
                f.write("-" * 80 + "\n")
                f.write(f"Instruction: {result['instruction']}\n")
                f.write(f"RAG Enhanced: {result.get('rag_enhanced', False)}\n")
                f.write(f"Strategy: {result.get('strategy_used', 'unknown')}\n")
                f.write(f"RAG Time: {result.get('rag_processing_time', 0):.3f}s\n")
                f.write(f"LLM Time: {result.get('llm_processing_time', 0):.3f}s\n")
                f.write(f"Total Time: {result.get('total_processing_time', 0):.3f}s\n")
                
                # Quality metrics for this question
                metrics = result.get('metrics', {})
                if metrics:
                    f.write("Quality Metrics:\n")
                    if 'rouge_l' in metrics:
                        f.write(f"  ROUGE-L: {metrics['rouge_l']:.3f}\n")
                    if 'bertscore_f1' in metrics:
                        f.write(f"  BERTScore F1: {metrics['bertscore_f1']:.3f}\n")
                
                f.write(f"\nExpected Output:\n{result['expected_output']}\n")
                f.write(f"\nGenerated Answer:\n{result.get('generated_answer', 'N/A')}\n")
                f.write("\n" + "=" * 80 + "\n")
        
        logger.info(f"Results saved to {filename}")
        
        # Also save JSON version for programmatic access
        json_filename = filename.replace('.txt', '.json')
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump({
                'summary_metrics': summary_metrics,
                'detailed_results': self.results,
                'test_info': {
                    'test_date': datetime.now().isoformat(),
                    'test_file': self.jsonl_path,
                    'llm_endpoint': self.llm_endpoint,
                    'total_questions': len(self.results)
                }
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"JSON results saved to {json_filename}")

async def main():
    """Main function to run the tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RAG + LLM system with JSONL data")
    parser.add_argument("--jsonl_path", type=str, default="src/data/data/splits/test.jsonl", 
                       help="Path to the JSONL test file")
    parser.add_argument("--llm_endpoint", type=str, default="http://localhost:8000/generate",
                       help="LLM API endpoint for generation")
    parser.add_argument("--max_questions", type=int, help="Maximum number of questions to test")
    parser.add_argument("--output_file", type=str, help="Output file name")
    
    args = parser.parse_args()
    
    # Create tester instance
    tester = RagLlmTester(args.jsonl_path, args.llm_endpoint)
    
    # Run tests
    await tester.run_all_tests(max_questions=args.max_questions)
    
    # Save results
    tester.save_results_to_file(args.output_file)
    
    # Print summary
    summary = tester.calculate_summary_metrics()
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total Questions: {summary.get('total_questions', 0)}")
    print(f"Coverage Rate: {summary.get('coverage_rate', 0):.2%}")
    print(f"Success Rate: {summary.get('success_rate', 0):.2%}")
    
    time_stats = summary.get('time_statistics', {})
    if 'avg_total_time' in time_stats:
        print(f"Average Total Time: {time_stats['avg_total_time']:.3f}s")
    if 'avg_rag_time' in time_stats:
        print(f"Average RAG Time: {time_stats['avg_rag_time']:.3f}s")
    if 'avg_llm_time' in time_stats:
        print(f"Average LLM Time: {time_stats['avg_llm_time']:.3f}s")
    
    if summary.get('quality_metrics'):
        quality = summary['quality_metrics']
        if 'avg_rouge_l' in quality:
            print(f"Average ROUGE-L: {quality['avg_rouge_l']:.3f}")
        if 'avg_bertscore_f1' in quality:
            print(f"Average BERTScore F1: {quality['avg_bertscore_f1']:.3f}")

if __name__ == "__main__":
    asyncio.run(main()) 