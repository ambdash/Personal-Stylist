from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import torch
import evaluate
import numpy as np
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelFinetuner:
    def __init__(self, base_model: str = "microsoft/phi-2"):
        self.base_model = base_model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = self._prepare_model()
        self.bert_score = evaluate.load("bertscore")

    def _prepare_model(self):
        # Load base model with 8-bit quantization
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            load_in_8bit=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # Prepare for LoRA fine-tuning
        model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=16,  # Rank
            lora_alpha=32,
            target_modules=["query_key_value"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        model = get_peft_model(model, lora_config)
        return model

    def _preprocess_data(self, examples: Dict) -> Dict:
        # Format the text for instruction fine-tuning
        texts = [
            f"Instruction: {example['instruction']}\nResponse: {example['response']}"
            for example in examples
        ]
        
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=512,
            padding="max_length"
        )
        
        return tokenized

    def train(self, dataset_path: str, output_dir: str):
        # Load and preprocess dataset
        dataset = load_dataset("json", data_files=dataset_path)
        tokenized_dataset = dataset.map(
            self._preprocess_data,
            batched=True,
            remove_columns=dataset["train"].column_names
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch"
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        )

        # Train
        trainer.train()
        
        # Save model
        trainer.save_model(output_dir)
        logger.info(f"Model saved to {output_dir}")

if __name__ == "__main__":
    finetuner = ModelFinetuner()
    finetuner.train(
        dataset_path="data/synthetic/fashion_qa.json",
        output_dir="models/finetuned"
    ) 