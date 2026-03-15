from typing import Literal
import torch
from .hook import Hook


class Interceptor(Hook):
    def __init__(self, model, start_layer: int = 0, end_layer: int = None, training_mode: Literal["prefill", "token"] = "token"):
        super().__init__(model)
        
        self.training_mode = training_mode
        self.activations = []
        self._capture_next = False  
        self._acts_buffer = {} # Utilisation d'un dict pour garantir l'ordre des layers
        
        self.start_layer = start_layer
        self.end_layer = end_layer if end_layer is not None else len(model.model.layers)
        
    def _get_layers_to_hook(self):
        # On passe None en data car on n'en a pas besoin pour juste intercepter
        return [(i, None) for i in range(self.start_layer, self.end_layer)]

    def _get_hook(self, layer_idx, data): # Ajout de 'data' pour respecter le contrat
        def _hook_fn(module, input, output):
            if not self._capture_next:
                return 
            
            # Détection du mode (Prompt vs Token)
            hidden_states = output[0] if isinstance(output, tuple) else output
            is_prefill = hidden_states.shape[1] > 1 #if prefill is true, we're in the prefill moment

            if self.training_mode == "token":
                if is_prefill:
                    return # ignore prefill moment
            else:
                if not is_prefill: #get only prefill
                    self._flush()
                    return   
                
            # Capture last token
            self._acts_buffer[layer_idx] = hidden_states[:, -1, :].detach() # stay on gpu
            
            # Si on a atteint la dernière couche, on range ça dans les activations finales
            if len(self._acts_buffer) == (self.end_layer - self.start_layer):
                self._flush()
                
        return _hook_fn
    
    def allow_one_capture(self):
        self._capture_next = True
        return self  
    
    def _flush(self, block_capture=True):
        if self._acts_buffer:
            # On trie par index de couche pour garantir l'ordre dans le stack
            sorted_layers = sorted(self._acts_buffer.keys())
            acts = [self._acts_buffer[l] for l in sorted_layers]
            
            stacked = torch.stack(acts, dim=0)  # [num_layers, batch, hidden_dim]
            stacked_cpu = stacked.permute(1, 0, 2).cpu()   # [batch, num_layers, hidden_dim] cpu
            
            for i in range(stacked_cpu.shape[0]):
                self.activations.append(stacked_cpu[i].unsqueeze(0)) #[1, num_layers, hidden_dim]
            
            self._acts_buffer = {}
            if block_capture:
                self._capture_next = False
    