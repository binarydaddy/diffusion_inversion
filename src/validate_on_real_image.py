
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
)

def main(args):
    manager = FluxModelManager(model_backend="diffusers", device=args.device, offload=args.offload, time_skipping_lora_path=args.timestep_skipping_lora_path)

    validation_result_dir = os.path.join(args.validataion_data_dir, f"validation_result_{args.timestep_to_skip_to}")
    os.makedirs(validation_result_dir, exist_ok=True)
    real_image_paths = sorted(glob.glob(os.path.join(args.validataion_data_dir, "*.png")))
    
    for image_data_path in tqdm(real_image_paths, desc="Processing generation data"):
        
        file_name = os.path.basename(image_data_path).replace(".png", "")
        prompt_path = os.path.join(args.validataion_data_dir, f"{file_name}.txt")

        with open(prompt_path, "r") as f:
            source_prompt = f.read()
            target_prompt = source_prompt

        validation_result = validate_on_single_example(manager, args.timestep_to_skip_to, source_prompt, target_prompt, image_data_path, args.device, args.offload)
        fname = os.path.basename(image_data_path).replace(".pt", "")
        torch.save(validation_result, os.path.join(validation_result_dir, f"{fname}_validation_result.pt"))

        # Save Images
        generation_from_inversion_image = validation_result["generation_from_inversion_result"]
        generation_from_inversion_with_lora_image = validation_result["generation_from_inversion_with_lora_result"]
        generation_from_inversion_image.save(os.path.join(validation_result_dir, f"{fname}_generation_from_inversion.png"))
        generation_from_inversion_with_lora_image.save(os.path.join(validation_result_dir, f"{fname}_generation_from_inversion_with_lora.png"))

        # Save Image Plot
        result_image = concatenate_validation_images(image_data_path, generation_from_inversion_image, generation_from_inversion_with_lora_image)
        result_image.save(os.path.join(validation_result_dir, f"{fname}_result.png"))

    return

def concatenate_validation_images(original_image, generation_from_inversion_image, generation_from_inversion_with_lora_image):
    # Create a figure with 3 subplots side by side, with extra height for titles
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    # Convert PIL images to numpy arrays if needed
    def prepare_image(img):
        if hasattr(img, 'numpy'):  # If it's a tensor
            return img.numpy()
        elif isinstance(img, Image.Image):  # If it's a PIL image
            return np.array(img)
        elif isinstance(img, str):
            img = Image.open(img)
            return np.array(img)
        else:
            return img
    
    images = [original_image, generation_from_inversion_image, generation_from_inversion_with_lora_image]
    titles = ["Original Image", "Inversion", "Inversion with LoRA"]
    
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

def validate_on_single_example(manager, timestep_to_skip_to, source_prompt, target_prompt, image_path, device="cuda", offload=True):

    inversion_result = manager.invert(
        image_path=image_path,
        source_prompt=source_prompt,
    )

    inversion_with_lora_result = manager.invert_with_timestep_skipping(
        image_path=image_path,
        source_prompt=source_prompt,
        timestep_to_skip_to=timestep_to_skip_to,
    )

    generation_from_inversion_result = manager.generate(
        start_latent=inversion_result.final_latent,
        prompt=target_prompt,
    )
    
    generation_from_inversion_with_lora_result = manager.generate(
        start_latent=inversion_with_lora_result.final_latent,
        prompt=target_prompt,
    )

    validation_result = {
        "source_prompt": source_prompt,
        "target_prompt": target_prompt,
        "timestep_to_skip_to": timestep_to_skip_to,
        "generation_from_inversion_result": generation_from_inversion_result.final_image,
        "generation_from_inversion_with_lora_result": generation_from_inversion_with_lora_result.final_image,
    }

    return validation_result
  
if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("--timestep_skipping_lora_path", type=str, default="/home/swhong/workspace/diffusion_inversion/trainer/flux_inversion_lora_target_0.3_512/checkpoint-3000/pytorch_lora_weights.safetensors")
    parser.add_argument("--timestep_to_skip_to", type=float, default=0.31)
    parser.add_argument("--validataion_data_dir", type=str, default="/home/swhong/workspace/diffusion_inversion/src/validation_real_image")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--offload", type=bool, default=True)
    args = parser.parse_args()
    
    main(args) 