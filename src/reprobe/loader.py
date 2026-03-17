import logging
from .steerer import Steerer
from .monitor import Monitor
from .probe import Probe
from pathlib import Path
from typing import Callable, Literal
import json
import os
import torch

logger = logging.getLogger(__name__)
class ProbeLoader:
    @staticmethod
    def from_registry(path: str) -> dict[int, Probe]:
        """
        Load from a registry.json
        Return {layer: Probe}
        """
        with open(path, "r") as f:
            registry = json.load(f)
        
        dir = os.path.dirname(path)
        probes = {
            "prefill": {},
            "token": {}
        }
        training_mode = registry["training_mode"]
        if not training_mode or training_mode not in ["prefill", "token", "all"]:
                raise ValueError(f"Registry has an invalid mode: {training_mode}. Must be between 'token', 'prefill' and 'all'")
                
        items = registry["probes"][training_mode].items() if training_mode != "all" else registry["probes"]["prefill"].items() + registry["probes"]["token"].items() 
        
        for key, meta in items:
            probe_path = os.path.join(dir, meta["filename"])
            probe = Probe.load_from_file(probe_path)
            mode = probe.meta["training_mode"]
            if not mode or mode not in ["prefill", "token"]:
                logger.warning(f"Probe layer {probe.meta["layer"]} of {probe.meta["model_id"]} has an invalid mode: {mode}. Must be between 'token' and 'prefill'. Skipped")
                continue
            probes[mode][probe.meta["layer"]] = probe
            
        return probes
    
    @staticmethod
    def from_file(path: str) -> dict[int, Probe]:
        """
        Load from a .pt file
        Return {layer: Probe}
        """
        content = torch.load(path)
        probes = {
            "prefill": {},
            "token": {}
        }
        training_mode = content["training_mode"]
        for key, data in content["probes"][training_mode].items():
            probe = Probe.load(
                data["state_dict"],
                mean_act=data["mean_act"],
                std_act=data["std_act"],
                **data["meta"]
            )
            mode = probe.meta["training_mode"]
            if not mode or mode not in ["prefill", "token"]:
                logger.warning(f"Probe layer {probe.meta["layer"]} of {probe.meta["model_id"]} has an invalid mode: {mode}. Must be between 'token' and 'prefill'. Skipped")
                continue
            probes[mode][probe.meta["layer"]] = probe
            
        return probes
    
    @staticmethod
    def load(path: str) -> dict[int, Probe]:
        path = Path(path)

        if path.suffix == ".pt":
            return ProbeLoader.from_file(path)

        if path.suffix == ".json":
            return ProbeLoader.from_registry(path)

        raise ValueError(f"Unsupported file type: {path}")

    
    @staticmethod
    def monitor(
        model, 
        path: str,
        filter: Callable[[dict], bool] = None
    ):
        probes = ProbeLoader.load(path)
        if filter:
            probes = {k: v for k, v in probes.items() if filter(v.meta)}
        
        return Monitor(model, list(probes.values()))
    
    @staticmethod
    def steerer(
        model,
        path: str,
        mode: Literal['projected', 'uniform'] = "projected",
        alpha: float | dict[int, float] | Callable[[dict], float] = 1.0,
        filter: Callable[[dict], bool] = None,
    ):
        probes = ProbeLoader.load(path)
        if filter:
            probes = {k: v for k, v in probes.items() if filter(v.meta)}
        
        if callable(alpha):
            probe_list = [(p, alpha(p.meta)) for p in probes.values()]
        elif isinstance(alpha, dict):
            probe_list = [(p, alpha.get(layer, 20.0)) for layer, p in probes.items()]
        else:
            probe_list = list(probes.values()) #Steerer will automatically add alpha
            
        return Steerer(model, probe_list, mode=mode, alpha=alpha) #alpha will be automatically ignored in the 2 first case