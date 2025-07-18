from flux_model_manager import FluxModelManager
from utils import get_aligned_generation_intermediate_latents, get_timestep_list_from_intermediate_latents
import torch
import json

p = "/home/swhong/workspace/diffusion_inversion/src/validation_no_train"
current_sample_name = "000000"
generation_data_path = f"{p}/{current_sample_name}_data.pt"

manager = FluxModelManager(model_backend="diffusers", device="cuda", offload=True)
generation_data = torch.load(generation_data_path, weights_only=False)

with open(f"{p}/{current_sample_name}_metadata.json", "r") as f:
    generation_metadata = json.load(f)

timestep_to_skip_to = 0.31

aligned_intermediate_latents = get_aligned_generation_intermediate_latents(generation_data)
timesteps = get_timestep_list_from_intermediate_latents(aligned_intermediate_latents)

velocity = aligned_intermediate_latents[1.0] - generation_data['final_latent']

inversion_result = manager.invert_with_straight_timestep_skipping(
    start_latent=generation_data['final_latent'],
    source_prompt=generation_metadata['prompt'],
    timestep_to_skip_to=timestep_to_skip_to,
    gt_velocity=velocity,
)

generation_result = manager.generate(
    prompt=generation_metadata['prompt'],
    start_latent=inversion_result.final_latent,
)

generation_result.final_image.save("generation_result.png")

