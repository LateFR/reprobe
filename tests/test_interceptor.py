import torch
from reprobe import Interceptor
import torch

def test_flush_layer_order():
    hidden_dim = 16
    batch = 5
    ids = [0.9678, 1.87]
    mock_model = {"model": {"layers": [i for i in range(200)]}}
    interceptor = Interceptor(mock_model, end_layer=200)

    # prefill
    interceptor._acts_buffer[15] = torch.full((batch, hidden_dim), ids[0])
    interceptor._acts_buffer[12] = torch.full((batch, hidden_dim), ids[1])
    interceptor._flush("prefill")

    # token
    interceptor._acts_buffer[15] = torch.full((batch, hidden_dim), ids[0])
    interceptor._acts_buffer[12] = torch.full((batch, hidden_dim), ids[1])
    interceptor._flush("token")

    acts = interceptor.finalize()
    
    for key in ["prefill", "token"]:
        assert torch.allclose(acts[key][0, 0, :], torch.full((hidden_dim,), ids[1]))
        assert torch.allclose(acts[key][0, 1, :], torch.full((hidden_dim,), ids[0]))
        
def test_interceptor_explicit_end_layer():
    target_end_layer = 200
    
    class FakeLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def register_forward_hook(self, fn): pass
    
    class FakeModel:
        class model:
            layers = torch.nn.ModuleList([FakeLayer() for _ in range(target_end_layer)])
    interceptor = Interceptor(FakeModel(), end_layer=target_end_layer)
    interceptor.attach()
    
    assert interceptor.end_layer == target_end_layer
    