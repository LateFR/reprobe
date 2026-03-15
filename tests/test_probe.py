import torch
import tempfile, os
from reprobe import Probe

def test_probe_save_and_load():
    # Mock probe
    probe = Probe(hidden_dim=16, concepts=["toxicity"], layer=5, model_id="test", training_mode="prefill")
    probe.mean_act = torch.zeros(16)
    probe.std_act = torch.ones(16)

    # On la sauvegarde dans un fichier temporaire
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name

    try:
        probe.training_mode = "prefill"
        probe.save(path)
        loaded = Probe.load_from_file(path)

        # Est-ce que les métadonnées survivent ?
        assert loaded.meta["layer"] == 5
        assert loaded.meta["hidden_dim"] == 16

        # Est-ce que les poids sont identiques ?
        original_w = probe.model[0].weight.data
        loaded_w = loaded.model[0].weight.data
        assert torch.allclose(original_w, loaded_w), "Les poids ont changé après save/load !"
    finally:
        os.remove(path)