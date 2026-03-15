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
    
class DatasetConfig(TypedDict):
    type: Literal["huggingface", "json"]
    name: str
    fn_data: Callable
    params: dict
    
class DataManager():
    def __init__(self, datasets: list[DatasetConfig]):
        self.datasets = datasets
        
    def load(self):
        datas = []
        for dataset in self.datasets:
            cfg = dataset.copy()
            fn = cfg.pop("fn_data")
            if cfg["type"] == "huggingface":
                ds = load_dataset(cfg.pop("name"), **cfg["params"])
                datas.append(fn(ds))