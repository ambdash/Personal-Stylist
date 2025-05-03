from data.generate_synthetic import SyntheticDataGenerator
from ml.training.finetune import ModelFinetuner
from ml.evaluation.compare_models import ModelEvaluator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_pipeline(questions: list):
    # Step 1: Generate synthetic data
    logger.info("Generating synthetic data...")
    generator = SyntheticDataGenerator()
    generator.generate_dataset(questions, "data/synthetic/fashion_qa.json")
    
    # Step 2: Fine-tune model
    logger.info("Fine-tuning model...")
    finetuner = ModelFinetuner()
    finetuner.train(
        dataset_path="data/synthetic/fashion_qa.json",
        output_dir="models/finetuned"
    )
    
    # Step 3: Evaluate models
    logger.info("Evaluating models...")
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_models(questions)
    
    logger.info(f"Pipeline completed. Average BERT Score: {results['average_bert_score']}")

if __name__ == "__main__":
    questions = [
        "Подбери стильный образ в стиле Y2K",
        # ... your questions list ...
    ]
    
    run_pipeline(questions) 