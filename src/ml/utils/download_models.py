from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model(model_name: str, save_dir: str = "models/pretrained"):
    """
    Download model and tokenizer for offline use.
    
    Args:
        model_name: HuggingFace model name
        save_dir: Directory to save the model
    """
    save_path = Path(save_dir) / model_name.split('/')[-1]
    os.makedirs(save_path, exist_ok=True)
    
    logger.info(f"Downloading {model_name} to {save_path}")
    
    # Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_path)
    logger.info(f"Tokenizer saved to {save_path}")
    
    # Download model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.save_pretrained(save_path)
    logger.info(f"Model saved to {save_path}")
    
    return str(save_path)

if __name__ == "__main__":
    models = [
        "microsoft/phi-2",
        "IlyaGusev/saiga_7b_lora",
        "IlyaGusev/saiga_mistral_7b",
    ]
    
    for model in models:
        try:
            path = download_model(model)
            print(f"Successfully downloaded {model} to {path}")
        except Exception as e:
            print(f"Failed to download {model}: {str(e)}") 