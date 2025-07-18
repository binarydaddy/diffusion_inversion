
import glob
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm
from flux_model_manager import FluxModelManager
from utils import (
    get_timestep_list_from_intermediate_latents,
    get_aligned_generation_intermediate_latents,
    pack,
    unpack
)

def main(args):
    manager = FluxModelManager(model_backend="diffusers", device=args.device, offload=args.offload, time_skipping_lora_path=args.timestep_skipping_lora_path)

    validation_result_dir = os.path.join(args.validated_result_dir, f"validation_result_{args.timestep_to_skip_to}")
    os.makedirs(validation_result_dir, exist_ok=True)
    
    generation_data_files = sorted(glob.glob(os.path.join(args.generation_data_dir, "*.pt"), recursive=True))
    
    for generation_data_dir in tqdm(generation_data_files, desc="Processing generation data"):
        generation_data = torch.load(generation_data_dir, weights_only=False)
        target_prompt = source_prompt = generation_data["metadata"]["prompt"]

        performance_dict, validation_result = validate_on_single_example(manager, args.timestep_to_skip_to, source_prompt, target_prompt, generation_data, args.device, args.offload)
        fname = os.path.basename(generation_data_dir).replace(".pt", "")
        torch.save(validation_result, os.path.join(validation_result_dir, f"{fname}_validation_result.pt"))

        # Save Images
        generation_from_inversion_image = validation_result["generation_from_inversion_result"]
        generation_from_inversion_with_lora_image = validation_result["generation_from_inversion_with_lora_result"]
        generation_from_inversion_image.save(os.path.join(validation_result_dir, f"{fname}_generation_from_inversion.png"))
        generation_from_inversion_with_lora_image.save(os.path.join(validation_result_dir, \
            f"{fname}_generation_from_inversion_with_lora.png"))
        generation_from_inversion_with_straight_timestep_skipping_image = validation_result["generation_from_inversion_with_straight_timestep_skipping_result"]
        generation_from_inversion_with_straight_timestep_skipping_image.save(os.path.join(validation_result_dir, \
             f"{fname}_generation_from_inversion_with_straight_timestep_skipping.png"))

        # Save Image Plot
        result_image = concatenate_validation_images(generation_data["final_image"], generation_from_inversion_image, \
             generation_from_inversion_with_lora_image, generation_from_inversion_with_straight_timestep_skipping_image)
        result_image.save(os.path.join(validation_result_dir, f"{fname}_result.png"))

    return

def concatenate_validation_images(original_image, generation_from_inversion_image, generation_from_inversion_with_lora_image, generation_from_inversion_with_straight_timestep_skipping_image):
    # Create a figure with 3 subplots side by side, with extra height for titles
    fig, axes = plt.subplots(1, 4, figsize=(26, 8))
    
    # Convert PIL images to numpy arrays if needed
    def prepare_image(img):
        if hasattr(img, 'numpy'):  # If it's a tensor
            return img.numpy()
        elif isinstance(img, Image.Image):  # If it's a PIL image
            return np.array(img)
        else:
            return img
    
    images = [original_image, generation_from_inversion_image, generation_from_inversion_with_lora_image, generation_from_inversion_with_straight_timestep_skipping_image]
    titles = ["Original Image", "Inversion", "Inversion with LoRA", "Inversion with Straight Timestep Skipping"]
    
    for i, (img, title) in enumerate(zip(images, titles)):
        img_array = prepare_image(img)
        axes[i].imshow(img_array)
        axes[i].set_title(title, fontsize=12, fontweight='bold', pad=10)
        axes[i].axis('off')  # Remove axes for cleaner look
    
    # Add padding at the top to prevent title cutoff
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(top=0.85)
    
    # Convert the plot to a PIL Image
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    result_image = Image.frombuffer('RGBA', fig.canvas.get_width_height(), buf, 'raw', 'RGBA', 0, 1)
    result_image = result_image.convert('RGB')  # Convert to RGB if needed
    
    plt.close(fig)  # Close the figure to free memory
    
    return result_image

def validate_on_single_example(manager, timestep_to_skip_to, source_prompt, target_prompt, generation_data, device="cuda", offload=True):
    generated_image = generation_data["final_latent"]

    # 1. Compare Lora Inversion vs Regular Inversion Performance Difference

    inversion_result = manager.invert(
        start_latent=generated_image,
        source_prompt=source_prompt,
    )

    inversion_with_lora_result = manager.invert_with_timestep_skipping(
        start_latent=generated_image,
        source_prompt=source_prompt,
        timestep_to_skip_to=timestep_to_skip_to,
    )
    
    generation_data['initial_latent'] = pack(generation_data['initial_latent'], 1024, 1024)
    gt_velocity = generation_data['initial_latent'] - generation_data['final_latent']
    inversion_with_straight_timestep_skipping_result = manager.invert_with_straight_timestep_skipping(
        start_latent=generated_image,
        source_prompt=source_prompt,
        timestep_to_skip_to=timestep_to_skip_to,
        gt_velocity=gt_velocity,
    )
    
    aligned_intermediate_latents = get_aligned_generation_intermediate_latents(generation_data)
    timesteps = get_timestep_list_from_intermediate_latents(aligned_intermediate_latents)

    performance_dict = {}
    # Compare Results with Generation GT Data.
    for timestep in timesteps:

        if timestep not in inversion_with_lora_result.intermediate_latents:
            continue

        gt_latent = aligned_intermediate_latents[timestep]
        lora_inversion_latent = inversion_with_lora_result.intermediate_latents[timestep]
        regular_inversion_latent = inversion_result.intermediate_latents[timestep]
        straight_timestep_skipping_latent = inversion_with_straight_timestep_skipping_result.intermediate_latents[timestep]

        lora_inversion_l2_distance = torch.norm(lora_inversion_latent - gt_latent)
        regular_inversion_l2_distance = torch.norm(regular_inversion_latent - gt_latent)
        straight_timestep_skipping_l2_distance = torch.norm(straight_timestep_skipping_latent - gt_latent)

        performance_dict[timestep] = {
            "lora_inversion_l2_distance": lora_inversion_l2_distance,
            "regular_inversion_l2_distance": regular_inversion_l2_distance,
            "straight_timestep_skipping_l2_distance": straight_timestep_skipping_l2_distance,
        }

    # Generate from inverted results
    generation_from_inversion_result = manager.generate(
        start_latent=inversion_result.final_latent,
        prompt=target_prompt,
    )

    generation_from_inversion_with_lora_result = manager.generate(
        start_latent=inversion_with_lora_result.final_latent,
        prompt=target_prompt,
    )

    generation_from_inversion_with_straight_timestep_skipping_result = manager.generate(
        start_latent=inversion_with_straight_timestep_skipping_result.final_latent,
        prompt=target_prompt,
    )

    validation_result = {
        "source_prompt": source_prompt,
        "target_prompt": target_prompt,
        "timestep_to_skip_to": timestep_to_skip_to,
        "performance_dict": performance_dict,
        "generation_from_inversion_result": generation_from_inversion_result.final_image,
        "generation_from_inversion_with_lora_result": generation_from_inversion_with_lora_result.final_image,
        "generation_from_inversion_with_straight_timestep_skipping_result": generation_from_inversion_with_straight_timestep_skipping_result.final_image,
    }

    return performance_dict, validation_result
  
if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("--timestep_skipping_lora_path", type=str, default="/home/swhong/workspace/diffusion_inversion/trainer/flux_inversion_lora_target_0.3_512/checkpoint-3000/pytorch_lora_weights.safetensors")
    parser.add_argument("--timestep_to_skip_to", type=float, default=0.31)
    parser.add_argument("--generation_data_dir", type=str, default="/home/swhong/workspace/diffusion_inversion/validation_data")
    parser.add_argument("--validated_result_dir", type=str, default="/home/swhong/workspace/diffusion_inversion/validation_result")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--offload", type=bool, default=True)
    args = parser.parse_args()
    
    main(args) 