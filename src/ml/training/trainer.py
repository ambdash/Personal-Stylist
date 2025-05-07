from transformers import TrainingArguments
from pathlib import Path
from src.ml.config import TrainingConfig

class ModelTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        
    def train(self, dataset_path: Path):
        training_args = TrainingArguments(
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            
            # Learning rate scheduling
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type="cosine",
            weight_decay=self.config.weight_decay,            
            max_grad_norm=self.config.max_grad_norm,
            fp16=self.config.fp16,
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps
        ) 