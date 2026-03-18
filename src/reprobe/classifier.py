from typing import Literal, TypedDict, Callable
from datasets import load_dataset

class Classifier():
    """
    Contract : classify(text) -> torch.Tensor shape of [batch], valeurs in [0, 1]
    normalize_output must respect this contract.
    """
    def __init__(self, model, device, normalize_output: None | Callable = None, tokenizer = None):
        self.model = model.to(device)
        self.normalize_output = normalize_output
        self.tokenizer = tokenizer
        self.device = device
        
    def classify(self, text):
        if self.tokenizer:
            encoded = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        else:
            encoded = self.model.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        out = self.model(**encoded)
        results = self.normalize_output(out) if (self.normalize_output) else out
        
        return results