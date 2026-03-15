from typing import Literal
import torch
from .hook import Hook


class Interceptor(Hook):
    def __init__(self, model, start_layer: int = 0, end_layer: int = None, training_mode: Literal["prefill", "token", "all"] = "prefill"):
        super().__init__(model)
        
        self.training_mode = training_mode
        self._activations = {
            "prefill": [],
            "token": []
        }
        self._capture_next = False  
        self._acts_buffer = {} # Utilisation d'un dict pour garantir l'ordre des layers
        
        self.start_layer = start_layer
        self.end_layer = end_layer if end_layer is not None else len(model.model.layers)
        
        self._was_prefill = True
        
    def _get_layers_to_hook(self):
        # None because we don't need special data
        return [(i, None) for i in range(self.start_layer, self.end_layer)]

    def _get_hook(self, layer_idx, data):
        def _hook_fn(module, input, output):
            if not self._capture_next:
                return 
            
            hidden_states = output[0] if isinstance(output, tuple) else output
            is_prefill = hidden_states.shape[1] > 1 #if prefill is true, we're in the prefill moment
            inference_moment = "prefill" if is_prefill else "token"
            
            if self.training_mode == "token":
                if is_prefill:
                    return # ignore prefill moment
            elif self.training_mode == "prefill":
                if not is_prefill: #get only prefill
                    self._flush("prefill")
                    return   
            elif self.training_mode == "all":
                if self._was_prefill and not is_prefill:
                    self._flush("prefill", block_capture=False) # flush to the prefill
                self._was_prefill = False
            # Capture last token
            self._acts_buffer[layer_idx] = hidden_states[:, -1, :].detach() # stay on gpu
            
            # If we have reached the last layer, we classify this as a final activation.
            if len(self._acts_buffer) == (self.end_layer - self.start_layer):
                if self.training_mode == "prefill":
                    self._flush("prefill", block_capture=True)
        return _hook_fn
    
    def allow_one_capture(self):
        self._capture_next = True
        self._was_prefill = True
        return self  
    
    def _flush(self, to: Literal["prefill", "token"] ,block_capture=True):
        if self._acts_buffer:
            # We sort by layer index to guarantee order in the stack
            sorted_layers = sorted(self._acts_buffer.keys())
            acts = [self._acts_buffer[l] for l in sorted_layers]
            
            stacked = torch.stack(acts, dim=0)  # [num_layers, batch, hidden_dim]
            stacked_cpu = stacked.permute(1, 0, 2).cpu()   # [batch, num_layers, hidden_dim] to cpu
            
            for i in range(stacked_cpu.shape[0]):
                self._activations[to] = [stacked_cpu[i].unsqueeze(0)] #[1, num_layers, hidden_dim]
            
            self._acts_buffer = {}
            if block_capture:
                self._capture_next = False
    
    
    def finalize(self):
        if self.training_mode in ("token", "all"):
            self._flush("token", block_capture=True)
        result = {}
        for key, acts in self._activations.items():
            result[key] = torch.cat(acts) if acts else None
        return result