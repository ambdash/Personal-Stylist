import json
from datasets import Dataset
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

def load_fashion_qa(json_path):
    """Load and clean fashion QA dataset"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    # Clean up the data
    cleaned_data = []
    for item in data:
        # Clean instruction and output
        instruction = item['instruction'].strip()
        output = item['output'].strip()
        
        # Remove instruction from output if it's repeated
        if output.startswith(instruction):
            output = output[len(instruction):].lstrip(" .:-\n")
        
        # Remove numbering if present
        if output.startswith(('1.', '2.', '3.', '4.', '5.')):
            output = output[2:].lstrip()
        
        # Remove markdown formatting if present
        output = output.replace('**', '').strip()
        
        cleaned_data.append({
            "instruction": instruction,
            "output": output
        })
    
    return cleaned_data

def preprocess_for_instruction_tuning(data):
    """Format data for instruction tuning"""
    processed_data = []
    for item in data:
        # Format the prompt with system message and instruction
        prompt = f"Вопрос: {item['instruction']}\nОтвет:"
        
        processed_data.append({
            "input_ids": prompt,
            "labels": item["output"]
        })
    return processed_data

def split_and_prepare_dataset(
    json_path, 
    train_size=0.8,
    val_size=0.1,
    test_size=0.1,
    random_state=42
):
    """
    Split and prepare dataset for training, validation, and testing.
    
    Args:
        json_path: Path to the JSON data file
        train_size: Proportion of data for training (default: 0.8 or 80%)
        val_size: Proportion of data for validation (default: 0.1 or 10%)
        test_size: Proportion of data for testing (default: 0.1 or 10%)
        random_state: Random seed for reproducibility
    
    Note on BERTScore:
        The evaluation uses BERTScore, which measures semantic similarity between
        generated and reference texts using BERT embeddings. It's particularly
        useful for tasks like this because:
        1. It captures semantic similarity better than exact matches
        2. It handles paraphrasing well
        3. It correlates better with human judgments
        4. It's language-agnostic and works well with Russian
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-5, "Split proportions must sum to 1"
    
    logger.info(f"Loading data from {json_path}")
    data = load_fashion_qa(json_path)
    total_examples = len(data)
    
    # First split: separate training data
    train_data, temp_data = train_test_split(
        data,
        train_size=train_size,
        random_state=random_state
    )
    
    # Second split: divide remaining data into validation and test
    val_data, test_data = train_test_split(
        temp_data,
        test_size=test_size/(test_size + val_size),
        random_state=random_state
    )
    
    # Preprocess each split
    train_processed = preprocess_for_instruction_tuning(train_data)
    val_processed = preprocess_for_instruction_tuning(val_data)
    test_processed = preprocess_for_instruction_tuning(test_data)
    
    # Convert to HuggingFace Datasets
    train_dataset = Dataset.from_list(train_processed)
    val_dataset = Dataset.from_list(val_processed)
    test_dataset = Dataset.from_list(test_processed)
    
    # Log split information
    logger.info(f"Dataset split complete:")
    logger.info(f"Total examples: {total_examples}")
    logger.info(f"Training examples: {len(train_dataset)} ({len(train_dataset)/total_examples:.1%})")
    logger.info(f"Validation examples: {len(val_dataset)} ({len(val_dataset)/total_examples:.1%})")
    logger.info(f"Test examples: {len(test_dataset)} ({len(test_dataset)/total_examples:.1%})")
    
    # Log example format
    logger.info("\nExample format:")
    logger.info(f"Input: {train_processed[0]['input_ids']}")
    logger.info(f"Output: {train_processed[0]['labels']}")
    
    return train_dataset, val_dataset, test_dataset 