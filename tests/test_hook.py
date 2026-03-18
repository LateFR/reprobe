import torch
from reprobe import Interceptor


def test_hook_layers_path_manual():
    # Non standard architecture
    class FakeLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def register_forward_hook(self, fn): pass
    
    class FakeModel:
        custom = type('obj', (object,), {'blocks': torch.nn.ModuleList([FakeLayer() for _ in range(5)])})()
    
    interceptor = Interceptor(FakeModel(), _layers_path="custom.blocks")
    interceptor.attach()
    assert len(interceptor._layers) == 5