from flux_model_manager import FluxModelManager
import torch
import os
from PIL import Image
import numpy as np

# Sanity Check

# Initialize with default settings
manager = FluxModelManager(name="flux-dev", device="cuda", offload=False)

manager.sanity_check()

gen_result = manager.generate(
    prompt="A cat holding a sign that says hello world.", 
    width=1024, 
    height=1024, 
    num_steps=50, 
    guidance=3.5, 
    seed=42, 
    save_intermediates=True)

timesteps = gen_result.timesteps

inversion_starting_ts_idx_0 = manager.invert(
    source_prompt="A cat holding a sign that says hello world.",
    width=1024,
    height=1024,
    num_steps=50,
    guidance=3.5,
    start_latent=gen_result.final_latent,
)

inversion_starting_ts_idx_5 = manager.invert_image_starting_from_timestep(
    timestep_idx=5,
    inversion_data=gen_result,
    prompt="A cat holding a sign that says hello world."
)

inversion_starting_ts_idx_10 = manager.invert_image_starting_from_timestep(
    timestep_idx=10,
    inversion_data=gen_result,
    prompt="A cat holding a sign that says hello world."
)

inversion_starting_ts_idx_15 = manager.invert_image_starting_from_timestep(
    timestep_idx=15,
    inversion_data=gen_result,
    prompt="A cat holding a sign that says hello world."
)

inversion_starting_ts_idx_20 = manager.invert_image_starting_from_timestep(
    timestep_idx=20,
    inversion_data=gen_result,
    prompt="A cat holding a sign that says hello world."
)

inversion_starting_ts_idx_25 = manager.invert_image_starting_from_timestep(
    timestep_idx=25,
    inversion_data=gen_result,
    prompt="A cat holding a sign that says hello world."
)

inversion_starting_ts_idx_30 = manager.invert_image_starting_from_timestep(
    timestep_idx=30,
    inversion_data=gen_result,
    prompt="A cat holding a sign that says hello world."
)

manager.save_inversion_results(
    result=inversion_starting_ts_idx_0,
    output_dir="output",
    filename_prefix="inversion_starting_ts_idx_0"
)

manager.save_inversion_results(
    result=inversion_starting_ts_idx_5,
    output_dir="output",
    filename_prefix="inversion_starting_ts_idx_5"
)

manager.save_inversion_results(
    result=inversion_starting_ts_idx_10,
    output_dir="output",
    filename_prefix="inversion_starting_ts_idx_10"
)
manager.save_inversion_results(
    result=inversion_starting_ts_idx_15,
    output_dir="output",
    filename_prefix="inversion_starting_ts_idx_15"
)
manager.save_inversion_results(
    result=inversion_starting_ts_idx_20,
    output_dir="output",
    filename_prefix="inversion_starting_ts_idx_20"
)
manager.save_inversion_results(
    result=inversion_starting_ts_idx_25,
    output_dir="output",
    filename_prefix="inversion_starting_ts_idx_25"
)
manager.save_inversion_results(
    result=inversion_starting_ts_idx_30,
    output_dir="output",
    filename_prefix="inversion_starting_ts_idx_30"
)

