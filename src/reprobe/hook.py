import logging
import torch

logger = logging.getLogger(__name__)

_KNOWN_LAYER_PATHS = [
    "model.layers",       # Llama, Qwen, Mistral, Phi-3, Gemma, Falcon
    "transformer.h",      # GPT-2, BLOOM
    "gpt_neox.layers",    # GPT-NeoX, Pythia
    "model.decoder.layers", # OPT
]

class Hook():
    def __init__(self, model, _layers_path: str | None = None):
        self.handles = []
        self.model = model
        self._layers = None
        self._layers_path = _layers_path
    
    def _get_layers_to_hook(self):
        raise NotImplementedError
    
    @staticmethod
    def _resolve_layers(model, layers_path: str | None = None):
        if layers_path:
            obj = model
            for attr in layers_path.split("."):
                obj = getattr(obj, attr)
            return obj
        
        for path in _KNOWN_LAYER_PATHS:
            try:
                obj = model
                for attr in path.split("."):
                    obj = getattr(obj, attr)
                if isinstance(obj, torch.nn.ModuleList):
                    return obj
            except AttributeError:
                continue
        
        raise ValueError(
            "Could not auto-detect transformer layers. "
            "Pass the layer path manually via `_layers_path`, e.g. _layers_path='model.layers'. "
            f"Tried: {_KNOWN_LAYER_PATHS}"
        )
        
    def _resolve_layers_if_none(self):
        if self._layers is None:
            self._layers = self._resolve_layers(self.model, self._layers_path)
    def attach(self):
        self._resolve_layers_if_none()
        for layer_idx, data in self._get_layers_to_hook():
            handle = self._layers[layer_idx].register_forward_hook(
                self._get_hook(layer_idx, data)
            )
            self.handles.append(handle)

        return self
            
    def detach(self):
        for handle in self.handles:
            handle.remove()