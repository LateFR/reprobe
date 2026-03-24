from typing import Literal
import torch
from .hook import Hook

class Interceptor(Hook):
    def __init__(
        self, 
        model,
        start_layer: int = 0, 
        end_layer: int = None, 
        training_mode: Literal["prefill", "token", "all"] = "prefill",
        _layers_path: str | None = None
    ):
        super().__init__(model, _layers_path)
        
        self.training_mode = training_mode
        self._current_batch = {
            "prefill": [],
            "token": []
        }
        self._capture_next = False  
        self._acts_buffer = {}
        self._current_token_prompts = []
        
        self.start_layer = start_layer
        
        self.end_layer = end_layer
        
        
    def _get_layers_to_hook(self):
        # None because we don't need special data
        return [(i, None) for i in range(self.start_layer, self.end_layer)]

    def _get_hook(self, layer_idx, data):
        def _hook_fn(module, input, output):
            if not self._capture_next:
                return 
            
            hidden_states = output[0] if isinstance(output, tuple) else output
            is_prefill = hidden_states.shape[1] > 1 #if prefill is true, we're in the prefill moment
            
            if self.training_mode == "token":
                if is_prefill:
                    return # ignore prefill moment
            elif self.training_mode == "prefill":
                if not is_prefill: #get only prefill
                    self._flush("prefill")
                    return   
            # Capture last token
            self._acts_buffer[layer_idx] = hidden_states[:, -1, :].detach() # stay on gpu
            
            # If we have reached the last layer, we classify this as a final activation.
            if len(self._acts_buffer) == (self.end_layer - self.start_layer):
                if self.training_mode == "prefill":
                    self._flush("prefill", block_capture=True)
                    
                elif self.training_mode == "token":
                    self._flush("token", block_capture=False)
                    
                elif self.training_mode == "all":
                    if is_prefill:
                        self._flush("prefill", block_capture=False)
                    else:
                        self._flush("token", block_capture=False)
                        
        return _hook_fn
    
    def allow_one_capture(self, batch_size):
        self._capture_next = True
        self._current_batch = {"prefill": [], "token": [[] for _ in range(batch_size)]}
        return self
    
    def _flush(self, to: Literal["prefill", "token"] ,block_capture=True):
        if self._acts_buffer:
            # We sort by layer index to guarantee order in the stack
            sorted_layers = sorted(self._acts_buffer.keys())
            acts = [self._acts_buffer[l] for l in sorted_layers]
            
            stacked = torch.stack(acts, dim=0)  # [num_layers, batch, hidden_dim]
            stacked_cpu = stacked.permute(1, 0, 2).float().cpu()   # [batch, num_layers, hidden_dim] to cpu
            
            for i in range(stacked_cpu.shape[0]):
                if to == "token":
                    self._current_batch["token"][i].append(stacked_cpu[i])  # [num_layers, hidden_dim]
                else:
                    self._current_batch[to] += [stacked_cpu[i].unsqueeze(0)] #[1, num_layers, hidden_dim]
            
            self._acts_buffer = {}
            if block_capture:
                self._capture_next = False
    
    def flush_batch(self) -> dict[str, torch.Tensor | None]:
        """
        Return:
            A dict which look like this: 
            {
                "prefill": Tensor[batch, num_layers, hidden_dim],
                "token": list[Tensor[K_i, num_layers, hidden_dim]]  
            }
            
        """
        result = {}
        if self._current_batch["prefill"]:
            result["prefill"] = torch.cat(self._current_batch["prefill"])
        else:
            result["prefill"] = None
            
        if any(self._current_batch["token"]):
            result["token"] = [
                torch.stack(prompt_tokens)  # [K_i, num_layers, hidden_dim]
                for prompt_tokens in self._current_batch["token"]
                if prompt_tokens
            ]                              
        else:
            result["token"] = None
            
        self._current_batch = {"prefill": [], "token": [[] for _ in range(len(self._current_batch["token"]))]}
        self._capture_next = False
        
        return result
    
    def attach(self):
        self._resolve_layers_if_none()
        if self.end_layer is None:
            self.end_layer = len(self._layers)
        return super().attach()