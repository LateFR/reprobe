from typing import Literal
from regex import D
import torch
from .hook import Hook
from .probe import Probe

class Steerer(Hook):
    def __init__(
            self,
            model,
            probes: list,
            mode: Literal["projected", "uniform"] = "projected",
            alpha: float = 1
        ):
        super().__init__(model)
        self.probes: list[tuple[Probe, float]] = [
            (p, alpha) if isinstance(p, Probe) else (p[0], p[1])
            for p in probes
        ]
        self.mode = mode

    def _get_hook(self, layer_idx, data):
        probe, alpha = data
        
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        direction = probe.get_direction().to(device, dtype)
        
        def _hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            
            if self.mode == "projected":
                hidden = Steerer._apply_projection(hidden, direction, alpha)
            else:
                hidden = Steerer._apply_uniform(hidden, direction, alpha)
            
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            
            return hidden
        return _hook_fn

    def _apply_projection(hidden, direction, alpha):
        # Scalair product
        # (batch, seq, dim) @ (dim,) -> (batch, seq)
        dot_product = torch.matmul(hidden, direction)
                
        # (batch, seq, 1) * (dim,) -> (batch, seq, dim)
        projection = dot_product.unsqueeze(-1) * direction
        
        hidden = hidden - alpha * projection
        return hidden
    
    def _apply_uniform(hidden, direction, alpha):
        return hidden - alpha * direction
    
    def _get_layers_to_hook(self):
        return [(probe.meta["layer"], (probe, alpha)) for probe, alpha in self.probes]
    