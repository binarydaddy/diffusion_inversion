from flux_model_manager import FluxModelManager
import torch
import os
from PIL import Image
import numpy as np
from glob import glob

# Sanity Check

# Initialize with default settings
manager = FluxModelManager(name="flux-dev", device="cuda", offload=False)

seed = 42

generation_result = manager.generate(
    prompt="A cat holding a sign that says hello world.",
    width=1024,
    height=1024,
    num_steps=50,
    guidance=3.5,
    seed=seed,
)

directory = "output"
manager.save_generation_result(
    result=generation_result,
    output_dir=directory,
    filename_prefix="generation"
)

original_image = f"{directory}/generation_final.png"

# Inversion test
inversion_timestep_idx = [0, 5, 10, 15, 20, 25, 30]

for idx in inversion_timestep_idx:
    inversion_result = manager.invert_image_starting_from_timestep(
        timestep_idx=idx,
        inversion_data=generation_result,
        prompt="A cat holding a sign that says hello world.",
    )

    manager.save_inversion_results(
        result=inversion_result,
        output_dir=os.path.join(directory, "inverted"),
        filename_prefix=f"inversion_ts_{idx}"
    )

# Generation test

for f in glob(os.path.join(directory, "inverted", "*.pt")):
    
    fname = f.split("/")[-1].split(".")[0]
    data = torch.load(f)
    final_latent = data["final_latent"]

    generation_result = manager.generate(
        prompt="A cat holding a sign that says hello world.",
        width=1024,
        height=1024,
        num_steps=50,
        guidance=3.5,
        start_latent=final_latent,
    )

    manager.save_generation_result(
        result=generation_result,
        output_dir=os.path.join(directory, "generated"),
        filename_prefix=fname
    )