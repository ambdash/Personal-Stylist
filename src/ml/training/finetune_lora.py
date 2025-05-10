import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import load_dataset
from src.ml.utils.data_prep import load_fashion_qa, to_hf_dataset

MODEL_NAME = "Qwen/Qwen1.5-7B-Chat"  # or "mistralai/Mistral-7B-v0.1"
OUTPUT_DIR = "models/finetuned"
DATA_PATH = "data/data/fashion_qa.json"

def preprocess(example):
    # Format for instruction tuning
    prompt = f"Instruction: {example['instruction']}\nResponse:"
    return {
        "input_ids": tokenizer(prompt, truncation=True, max_length=512, padding="max_length")["input_ids"],
        "labels": tokenizer(example["output"], truncation=True, max_length=512, padding="max_length")["input_ids"]
    }

if __name__ == "__main__":
    data = load_fashion_qa(DATA_PATH)
    dataset = to_hf_dataset(data)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    dataset = dataset.map(preprocess)

    # Load model in 8bit for efficiency
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_8bit=True,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05,
        bias="none", task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR) 