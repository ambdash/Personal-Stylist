# Персональный стилист на основе LLM и графовой базы данных


Проект представляет собой систему персонального стилиста, основанную на языковых моделях (LLM), с возможностью тонкой настройки на данных о моде и стиле.


## Setup

1. Install Poetry:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies:
```bash
poetry install
```

3. Initialize wandb:
```bash
poetry run wandb login
```

## Data Structure

Place your fashion QA data in:
```
data/
  data/
    fashion_qa.json
```

The JSON file should have the following structure:
```json
[
  {
    "instruction": "Question about fashion",
    "output": "Answer about fashion"
  },
  ...
]
```

## Запуск

1. Обучить модель:
```bash
poetry run python src/ml/run.py \
  --mode train \
  --model "microsoft/phi-2" \
  --data_path "data/data/fashion_qa.json" \
  --output_dir "models/finetuned" \
  --wandb_project "llm-finetuning"
```

2. Оценить качество модели:
```bash
poetry run python src/ml/run.py \
  --mode test \
  --model "microsoft/phi-2" \
  --data_path "data/data/fashion_qa.json" \
  --output_dir "models/finetuned" \
  --wandb_project "llm-finetuning"
```

## Выбранные архитектуры

- microsoft/phi-2
- mistralai/Mistral-7B-v0.1
- IlyaGusev/saiga_llama3_8b
- IlyaGusev/saiga2_7b_lora
- IlyaGusev/saiga_mistral_7b

## Wandb Integration

Трекинг ML-экспериментов проводится в Weights&Biases (Wandb)