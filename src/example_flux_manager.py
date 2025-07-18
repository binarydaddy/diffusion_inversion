"""
Example usage of FluxModelManager for image inversion and generation.
"""

import argparse
import torch
from flux_model_manager import FluxModelManager


def example_inversion(manager, image_path, source_prompt, output_dir):
    """Example of using the manager for image inversion."""
    print("\n=== Image Inversion Example ===")
    
    # Perform inversion
    result = manager.invert_image(
        image_path=image_path,
        source_prompt=source_prompt,
        num_steps=50,
        guidance=3.5,
        save_intermediates=True
    )
    
    # Save the inversion results
    manager.save_inversion_results(
        result,
        output_dir,
        "example_inversion"
    )
    
    print(f"Inversion complete. Final latent shape: {result.final_latent.shape}")
    print(f"Collected {len(result.intermediate_latents)} intermediate latents")
    
    return result


def example_generation(manager, prompt, output_dir, starting_latent=None):
    """Example of using the manager for image generation."""
    print("\n=== Image Generation Example ===")
    
    # Generate image
    result = manager.generate(
        prompt=prompt,
        width=1024,
        height=1024,
        num_steps=50,
        guidance=3.5,
        seed=42,
        save_intermediates=True,
        starting_latent=starting_latent
    )
    
    # Save the results
    manager.save_generation_result(result, output_dir, "example_generation")
    
    print(f"Generation complete. Image size: {result.final_image.size}")
    print(f"Collected {len(result.intermediate_latents)} intermediate latents")
    
    return result


def example_inversion_to_generation(manager, image_path, source_prompt, target_prompt, output_dir):
    """Example of inverting an image and then generating with a different prompt."""
    print("\n=== Inversion-to-Generation Example ===")
    
    # First, invert the image
    inversion_result = manager.invert_image(
        image_path=image_path,
        source_prompt=source_prompt,
        num_steps=50,
        guidance=3.5,
        save_intermediates=True
    )
    
    # Save inversion data
    manager.save_inversion_results(
        inversion_result,
        output_dir,
        "inversion_to_gen_inversion"
    )
    
    # Then generate from the inverted latent with a new prompt
    result = manager.generate(
        prompt=target_prompt,
        width=inversion_result.metadata['width'],
        height=inversion_result.metadata['height'],
        num_steps=50,
        guidance=3.5,
        starting_latent=inversion_result.final_latent,
        save_intermediates=True
    )
    
    # Save generation results
    manager.save_generation_result(result, output_dir, "inversion_to_generation")
    
    print("Inversion-to-generation complete!")
    
    return result

def example_inversion_with_timestep_skipping(manager, image_path, source_prompt, target_prompt, output_dir, timestep_skipping_lora_path, timestep_to_skip_to):
    """Example of inverting an image with timestep skipping."""
    print("\n=== Inversion with Timestep Skipping Example ===")
    
    # First, invert the image with timestep skipping
    inversion_result = manager.invert_with_timestep_skipping(
        image_path=image_path,
        source_prompt=source_prompt,
        timestep_to_skip_to=timestep_to_skip_to,
    )

    manager.save_inversion_results(
        inversion_result,
        output_dir,
        "inversion_with_timestep_skipping_inv"
    )

    result = manager.generate(
        prompt=target_prompt,
        width=inversion_result.metadata['width'],
        height=inversion_result.metadata['height'],
        num_steps=50,
        guidance=3.5,
        start_latent=inversion_result.final_latent,
        save_intermediates=True
    )

    manager.save_generation_result(result, output_dir, "inversion_with_timestep_skipping_gen")

    print("Inversion with timestep skipping complete!")

    return result


def main():
    parser = argparse.ArgumentParser(description='FluxModelManager Examples')
    parser.add_argument('--example', type=str, choices=['inversion', 'generation', 'both', 'inversion_to_generation', "inversion_with_timestep_skipping"],
                        default='generation', help='Which example to run')
    parser.add_argument('--model', type=str, default='flux-dev', help='Model name')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--offload', action='store_true', help='Offload models to save memory')
    parser.add_argument('--image_path', type=str, help='Path to input image (for inversion)')
    parser.add_argument('--source_prompt', type=str, default='a photo', help='Source prompt for inversion')
    parser.add_argument('--prompt', type=str, default='a beautiful landscape with mountains and a lake',
                        help='Prompt for generation')
    parser.add_argument('--target_prompt', type=str, default='a painting in van gogh style',
                        help='Target prompt for inversion-to-generation')
    parser.add_argument('--output_dir', type=str, default='output_examples', help='Output directory')
    parser.add_argument('--timestep_skipping_lora_path', type=str, default=None, help='Path to timestep skipping LoRA weights')
    parser.add_argument('--timestep_to_skip_to', type=float, default=None, help='Timestep to skip to')
    
    args = parser.parse_args()
    
    # Initialize the model manager
    print("Initializing FluxModelManager...")
    manager = FluxModelManager(
        name=args.model,
        device=args.device,
        offload=args.offload,
        model_backend="diffusers",
        time_skipping_lora_path=args.timestep_skipping_lora_path
    )
    
    # Run the requested example
    if args.example == 'inversion':
        if not args.image_path:
            raise ValueError("Image path required for inversion example")
        example_inversion(manager, args.image_path, args.source_prompt, args.output_dir)
        
    elif args.example == 'generation':
        example_generation(manager, args.prompt, args.output_dir)
        
    elif args.example == 'both':
        # Run generation
        example_generation(manager, args.prompt, args.output_dir)
        
        # Run inversion if image provided
        if args.image_path:
            example_inversion(manager, args.image_path, args.source_prompt, args.output_dir)
        else:
            print("\nSkipping inversion example (no image path provided)")
            
    elif args.example == 'inversion_to_generation':
        if not args.image_path:
            raise ValueError("Image path required for inversion-to-generation example")
        example_inversion_to_generation(
            manager, args.image_path, args.source_prompt, args.target_prompt, args.output_dir
        )
    
    elif args.example == 'inversion_with_timestep_skipping':
        if not args.image_path:
            raise ValueError("Image path required for inversion-to-generation example")
        if not args.timestep_skipping_lora_path or not args.timestep_to_skip_to:
            raise ValueError("Timestep skipping LoRA path and timestep to skip to must be provided")
        
        example_inversion_with_timestep_skipping(
            manager, args.image_path, args.source_prompt, args.target_prompt, args.output_dir, args.timestep_skipping_lora_path, args.timestep_to_skip_to
        )
    
    print("\nAll examples completed!")


if __name__ == "__main__":
    main() 