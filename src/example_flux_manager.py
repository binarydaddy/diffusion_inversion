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
    inverted_latent, intermediate_data = manager.invert_image(
        image_path=image_path,
        source_prompt=source_prompt,
        num_steps=50,
        guidance=3.5,
        save_intermediates=True
    )
    
    # Save the intermediate data
    manager.save_intermediate_data(
        intermediate_data,
        f"{output_dir}/inversion_data.pt"
    )
    
    print(f"Inversion complete. Final latent shape: {inverted_latent.shape}")
    print(f"Collected {len(intermediate_data['latents'])} intermediate latents")
    
    return inverted_latent, intermediate_data


def example_generation(manager, prompt, output_dir, starting_latent=None):
    """Example of using the manager for image generation."""
    print("\n=== Image Generation Example ===")
    
    # Generate image
    result = manager.generate(
        prompt=prompt,
        width=1024,
        height=1024,
        num_steps=50,
        guidance=7.5,
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
    inverted_latent, inversion_data = manager.invert_image(
        image_path=image_path,
        source_prompt=source_prompt,
        num_steps=50,
        guidance=3.5,
        save_intermediates=True
    )
    
    # Save inversion data
    manager.save_intermediate_data(
        inversion_data,
        f"{output_dir}/inversion_to_gen_inversion_data.pt"
    )
    
    # Then generate from the inverted latent with a new prompt
    result = manager.generate(
        prompt=target_prompt,
        width=inversion_data['metadata']['width'],
        height=inversion_data['metadata']['height'],
        num_steps=50,
        guidance=7.5,
        starting_latent=inverted_latent,
        save_intermediates=True
    )
    
    # Save generation results
    manager.save_generation_result(result, output_dir, "inversion_to_generation")
    
    print("Inversion-to-generation complete!")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='FluxModelManager Examples')
    parser.add_argument('--example', type=str, choices=['inversion', 'generation', 'both', 'inversion_to_generation'],
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
    
    args = parser.parse_args()
    
    # Initialize the model manager
    print("Initializing FluxModelManager...")
    manager = FluxModelManager(
        name=args.model,
        device=args.device,
        offload=args.offload
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
    
    print("\nAll examples completed!")


if __name__ == "__main__":
    main() 