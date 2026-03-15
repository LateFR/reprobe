from typing import Literal
from detoxify import Detoxify
import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.compute_probs import Probe
from scripts.hook import Hook
from collections import defaultdict

class Listener(Hook):
    def __init__(self, model, probes: list[Probe]):
        super().__init__(model)
        self.probes = {p.meta["layer"]: p for p in probes}
        print(self.probes.keys())
        self.model = model
        self._current_step_data = {}
        self.history = []
    
    def _get_layers_to_hook(self):
        return self.probes.items() # layer_idx, probe
    
    def _get_hook(self, layer_idx, probe):
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        direction = probe.get_direction().to(device, dtype)
        
        mean_act = probe.mean_act.to(device, dtype)
        std_act = probe.std_act.to(device, dtype)
        
        mode = probe.meta["training_mode"]
        
        accepted_modes = ["prefill", "token"]
        if mode not in accepted_modes:
            raise ValueError(f"Probe must have training_mode beetween {accepted_modes}")
        def _hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            is_prefill = hidden.shape[1] > 1
            if mode == "prefill" and not is_prefill:
                return  # ignore les tokens générés
            elif mode == "token" and is_prefill:
                return  # ignore le prefill
                
            # we got the last token
            # Shape: [batch, seq_len, hidden_dim] -> [batch, hidden_dim]
            last_token_act = hidden[:, -1, :].detach()
            
            if layer_idx == min(self.probes.keys()) or len(self._current_step_data) == len(self.probes):
                self._flush_step()
                
                
            with torch.no_grad():
                #normalize
                token_normalized = (last_token_act - mean_act) / std_act
                # We project and then convert to probability (0 to 1).
                logits = torch.matmul(token_normalized, direction)
                prob = torch.sigmoid(logits).item()
            
            self._current_step_data[layer_idx] = prob
            
            
        return _hook_fn
    
    def _flush_step(self):
        if self._current_step_data:
            self.history.append(self._current_step_data)
            self._current_step_data = {}
            
    def get_history(self, flush_buffer = True):
        res = self.history.copy()
        
        if flush_buffer:
            self.flush_buffer()
        
        return res
    
    def score(self, strategy: Literal["max_of_means", "mean_of_means", "max_absolute"]="max_of_means", flush_buffer=True):
        self._flush_step()
        if not self.history:
            return 0.0
            
        steps = self.history.copy()
        if flush_buffer:
            self.flush_buffer()

        
        step_scores = [
            sum(layer_probs.values()) / len(layer_probs) 
            for layer_probs in steps 
            if layer_probs # security if empty step
        ]

        if strategy == "max_of_means":
            return max(step_scores) if step_scores else 0.0
        
        if strategy == "mean_of_means":
            return sum(step_scores) / len(step_scores) if step_scores else 0.0

        if strategy == "max_absolute":
            return max(max(layer_probs.values()) for layer_probs in steps)

    def flush_buffer(self):
        self.history = [] 
        self._current_step_data = {}
        
        

