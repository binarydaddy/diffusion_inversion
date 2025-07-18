from flux_model_manager import FluxModelManager
import torch
import os
from PIL import Image
import numpy as np

# Sanity Check

# Initialize with default settings
manager = FluxModelManager(name="flux-dev", device="cuda", offload=False)
manager.sanity_check()