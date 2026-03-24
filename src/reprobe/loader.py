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
                
        for mode in ["prefill", "token"]:
            for key, meta in registry["probes"][mode].items():
                probe_path = os.path.join(dir, meta["filename"])
                probe = Probe.load_from_file(probe_path)
                probe_mode = probe.meta["training_mode"]
                if not probe_mode or probe_mode not in ["prefill", "token"]:
                    logger.warning(f"Probe layer {probe.meta["layer"]} of {probe.meta["model_id"]} has an invalid mode: {probe_mode}. Must be between 'token' and 'prefill'. Skipped")
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
        for mode in ["prefill", "token"]:
            for key, data in content["probes"][mode].items():
                probe = Probe.load(
                    data["state_dict"],
                    mean_act=data["mean_act"],
                    std_act=data["std_act"],
                    **data["meta"]
                )
                probe_mode = probe.meta["training_mode"]
                if not probe_mode or probe_mode not in ["prefill", "token"]:
                    logger.warning(f"Probe layer {probe.meta["layer"]} of {probe.meta["model_id"]} has an invalid mode: {probe_mode}. Must be between 'token' and 'prefill'. Skipped")
                    continue
                probes[mode][probe.meta["layer"]] = probe
                
        return probes
    
    @staticmethod
    def load(path: str, **hf_kwargs) -> dict[int, Probe]:
        p = Path(path)

        if p.exists():
            if p.suffix == ".pt":
                return ProbeLoader.from_file(p)

            if p.suffix == ".json":
                return ProbeLoader.from_registry(p)

            raise ValueError(f"Unsupported file type: {p}")

        try:
            from huggingface_hub import hf_hub_download, list_repo_files
        except ImportError:
            raise ImportError(
                f"Path '{p}' not found locally. "
                "To load from HuggingFace, install huggingface_hub: pip install huggingface_hub"
            )
        
        try:
            files = [f for f in list_repo_files(path) if f.endswith(".pt")]
            if not files:
                raise ValueError(f"No .pt file found in HuggingFace repo '{path}'.")
            
            path = hf_hub_download(path, filename=files[0], **hf_kwargs)
        except:
            raise ValueError(f"Path doesn't exist and ins't available on hugging face: '{path}'.")
        return ProbeLoader.load(path)
    @staticmethod
    def _check_mode(
        mode: Literal["prefill", "token", "all", "auto"],
        probes: dict[str, Probe],
        return_flatten_probes = False
    ) -> dict[str, Probe] | list[Probe]:
        
        if mode in ["prefill", "token"]:
            if not probes[mode]:
                raise ValueError(f"The probes given are not compatible with the mode {mode}. Probes provided has the following keys: {probes.keys()}")
            if return_flatten_probes:
                return list(probes[mode].values())
            
        elif mode == "all":
            if (not "prefill" in probes.keys()) or (not "token" in probes.keys()):
                raise ValueError(f"The probes given are not compatible with the mode {mode}. Probes provided has the following keys: {probes.keys()}")
            if return_flatten_probes:
                return list(probes["prefill"].values()) + list(probes["token"].values())
            
        elif mode == "auto":
            probe_list = list(probes["prefill"].values()) + list(probes["token"].values())
            if not probe_list:
                raise ValueError("No probes found")
            if return_flatten_probes:
                return probe_list
            
        else:
            raise ValueError("Invalid mode")
        
        return probes # if return_flatten_probes=False
            
    @staticmethod
    def monitor(
        model, 
        path: str,
        mode: Literal["prefill", "token", "all", "auto"] = "auto",
        filter: Callable[[dict], bool] = None,
        _layers_path: str | None = None,
        **hf_kwargs,
    ):
        """
    Create a Monitor from a probe file.

    Args:
        model: The transformer model to attach hooks to.
        path: Path to a registry.json, a .pt probe file or a hugging face repository
        mode: Which probes to load.
            - "prefill": only prefill probes (raises if none found)
            - "token": only token probes (raises if none found)
            - "all": both prefill and token (raises if either is missing)
            - "auto": all available probes, no validation on mode coverage
        filter: Optional callable receiving probe meta dict, returns True to keep the probe.
            Example: filter=lambda meta: meta["layer"] in [12, 13, 14]
        _layers_path: Optional path to the transformer layers attribute, e.g. "model.layers".
                      Only needed for non-standard architectures not auto-detected by reprobe.
                      Example: _layers_path="custom.transformer.blocks"
        **hg_kwargs: Extra arguments passed to hf_hub_download (e.g. token, revision, cache_dir).
                     Only used when loading from HuggingFace Hub
    """
        probes = ProbeLoader.load(path, **hf_kwargs)
        probes = ProbeLoader._check_mode(mode, probes, return_flatten_probes=True)
        if filter:
            probes = [p for p in probes if filter(p.meta)]
        
        return Monitor(model, probes, _layers_path = _layers_path)
    
    @staticmethod
    def steerer(
        model,
        path: str,
        mode: Literal["prefill", "token", "all", "auto"] = "auto",
        steering_mode: Literal['projected', 'uniform'] = "projected",
        alpha: float | dict[int, float] | dict[str, float] | Callable[[dict], float] = 1.0,
        filter: Callable[[dict], bool] = None,
        _layers_path: str | None = None,
        **hf_kwargs
    ):
        """
        Create a Steerer from a probe file.

        Args:
            model: The transformer model to attach hooks to.
            path: Path to a registry.json or .pt probe file.
            mode: Which probes to load. See ProbeLoader.monitor() for details.
            steering_mode: Steering projection method.
                - "projected": subtracts only the component along the probe direction (recommended)
                - "uniform": subtracts the full direction vector scaled by alpha
            alpha: Steering strength. Accepts:
                - float: same alpha for all probes
                - dict[int, float]: per-layer alpha, e.g. {12: 0.5, 13: 1.0}
                - dict[str, float]: per-mode alpha, e.g. {"prefill": 0.7, "token": 1.0}
                - Callable[[dict], float]: receives probe meta, returns alpha.
                Covers any combination: lambda meta: 0.7 if meta["training_mode"] == "prefill" else 1.0
            filter: Optional callable receiving probe meta dict, returns True to keep the probe.
                Example: filter=lambda meta: meta["layer"] in [12, 13, 14]
            _layers_path: Optional path to the transformer layers attribute, e.g. "model.layers".
                        Only needed for non-standard architectures not auto-detected by reprobe.
                        Example: _layers_path="custom.transformer.blocks"
            **hg_kwargs: Extra arguments passed to hf_hub_download (e.g. token, revision, cache_dir).
                        Only used when loading from HuggingFace Hub
        """
        probes = ProbeLoader.load(path, **hf_kwargs)
        probes = ProbeLoader._check_mode(mode, probes, return_flatten_probes=True)
        if filter:
            probes = [p for p in probes if filter(p.meta)]
        
        if callable(alpha):
            probe_list = [(p, alpha(p.meta)) for p in probes]
            
        elif isinstance(alpha, dict):
            first_key = next(iter(alpha))
            if isinstance(first_key, str):  # mode dict
                probe_list = [(p, alpha.get(p.meta["training_mode"], 1.0)) for p in probes]
            else:  # layer dict
                probe_list = [(p, alpha.get(p.meta["layer"], 1.0)) for p in probes]
                
        else:
            probe_list = probes #Steerer will automatically add alpha
            
        return Steerer(model, probe_list, mode=steering_mode, alpha=alpha, _layers_path = _layers_path) #alpha will be automatically ignored in the 2 first case
    
    