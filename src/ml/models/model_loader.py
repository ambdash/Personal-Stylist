from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Tuple

class ModelLoader:
    @staticmethod
    def load_model() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        model_name = "microsoft/phi-2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float32
        )
        return model, tokenizer 