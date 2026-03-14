import os
import tempfile

import torch

from reprobe import ProbeLoader, ProbesTrainer




def compare_probes(orig, new):
    assert orig.meta == new.meta
    assert torch.allclose(orig.model[0].weight.data, new.model[0].weight.data)
    
    
def test_probe_and_load_abstractor():
    hidden_dim = 16
    size = 100
    acts = torch.full((size, 3, hidden_dim), 4).float()
    labels = torch.cat([torch.ones(int(size/2)), torch.zeros(int(size/2))])
    
    model_id = "test"
    trainer = ProbesTrainer(model_id, hidden_dim)
    trainer.train_probes(acts, labels, ["test"],  epochs=1)
    probes = trainer.probes
    with tempfile.TemporaryDirectory() as tmpdir:
        
        trainer.save(tmpdir, one_file=True)
        
        loaded = ProbeLoader.from_file(os.path.join(tmpdir, f"{model_id}_probes.pt"))
        for layer in probes:
            orig = probes[layer]
            new = loaded[layer]
            compare_probes(orig, new)
        
        trainer.save(tmpdir)
        
        loaded = ProbeLoader.from_registry(os.path.join(tmpdir, "registry.json"))
        for layer in probes:
            orig = probes[layer]
            new = loaded[layer]
            compare_probes(orig, new)
    