import os
import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import json
from einops import rearrange

from flux.sampling import denoise, get_schedule, prepare, unpack
from flux.util import load_ae, load_clip, load_flow_model, load_t5, configs

from flux_model_manager import FluxModelManager

class InversionAnalyzer:
    """
    Manager class for Flux diffusion models.
    Handles model loading, image inversion, and generation with intermediate data collection.
    """
    
    def __init__(self, flux_model_manager: FluxModelManager, device: str = "cuda"):
        """
        Initialize the model manager with specified configuration.
        
        Args:
            name: Model name (e.g., "flux-dev", "flux-schnell")
            device: Device to load models on
            offload: Whether to offload models to CPU to save memory
        """
        
        self.device = device
        self.torch_device = torch.device(device)
        self.flux_model_manager = flux_model_manager
    
    def run_inversion_starting_from_timestep(self, image_path: str, prompt: str, num_steps: int, guidance: float, seed: int, start_latent: torch.Tensor, start_timestep: int):
        pass


    