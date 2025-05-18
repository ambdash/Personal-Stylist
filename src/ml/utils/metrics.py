from typing import Dict
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import nltk
from bert_score import score
import torch

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def calculate_metrics(generated: str, reference: str) -> Dict[str, float]:
    """Calculate evaluation metrics between generated and reference text."""
    # Tokenize for BLEU
    gen_tokens = word_tokenize(generated.lower())
    ref_tokens = word_tokenize(reference.lower())
    
    # Calculate BLEU
    bleu = sentence_bleu([ref_tokens], gen_tokens)
    
    # Calculate BERTScore
    P, R, F1 = score([generated], [reference], lang="ru", verbose=False)
    
    return {
        "bleu": float(bleu),
        "bert_score_precision": float(P.mean()),
        "bert_score_recall": float(R.mean()),
        "bert_score_f1": float(F1.mean())
    } 