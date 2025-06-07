#!/usr/bin/env python3

import sys
sys.path.append('.')

from scripts.enhanced_text_processor import EnhancedTextProcessor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_examples():
    """Test the improved processor with various examples"""
    processor = EnhancedTextProcessor()
    
    test_cases = [
        # Original example - should only show обувь
        "Какую летнюю обувь лучше выбрать для офиса в 2025 году?",
        
        # Theater example - should filter out informal styles
        "Какие модные варианты одежды подойдут для посещения театра зимой, чтобы выглядеть эффектно?",
        
        # Dinner example - should be more relevant
        "Я ищу подходящий стиль для ужина на набережной в теплую погоду.",
        
        # Styling context examples
        "Как стилизовать свитер для офиса?",
        "С чем носить розовый пиджак?",
        "Как стильно сочетать джемпер свободного кроя?",
        
        # Style-only example (should work with just keynodes)
        "Что надеть в стиле y2k?",
        
        # Specific item with multiple keynodes
        "Какую сумку выбрать для деловых встреч зимой?"
    ]
    
    try:
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{'='*80}")
            print(f"TEST CASE {i}: {test_case}")
            print('='*80)
            
            final_prompt, processing_time = processor.process_text(test_case)
            
            print(f"\nRESULT:")
            print(final_prompt)
            print(f"\nProcessing time: {processing_time:.2f} seconds")
            print("-" * 80)
    
    finally:
        processor.close()

if __name__ == "__main__":
    test_examples() 