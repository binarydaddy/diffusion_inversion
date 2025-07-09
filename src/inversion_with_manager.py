"""
Modified version of inversion_re.py that uses FluxModelManager.
This demonstrates how to integrate the manager class into existing code.
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
import json

from flux_model_manager import FluxModelManager
from inversion_re import (
    SamplingOptions, 
    analyze_single_step_denoising,
    analyze_multi_step_denoising,
    compare_denoising_strategies,
    render_latents_and_compute_psnr_and_compute_likelihood
)


def run_inversion_with_manager(args):
    """Run inversion using FluxModelManager."""
    
    # Initialize the model manager
    manager = FluxModelManager(
        name=args.name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        offload=args.offload
    )
    
    # Perform inversion
    inverted_latent, inversion_data = manager.invert_image(
        image_path=args.source_img_dir,
        source_prompt=args.source_prompt,
        num_steps=args.num_steps,
        guidance=args.guidance,
        seed=None,  # Random seed
        order=args.order,
        save_intermediates=True,
        feature_path=args.feature_path,
        inject_step=args.inject
    )
    
    # Now perform forward generation with target prompt
    print("\nPerforming forward generation with target prompt...")
    
    # Get image dimensions from inversion metadata
    width = inversion_data['metadata']['width']
    height = inversion_data['metadata']['height']
    
    # Generate from inverted latent with target prompt
    generation_result = manager.generate(
        prompt=args.target_prompt,
        width=width,
        height=height,
        num_steps=args.num_steps,
        guidance=args.guidance,
        starting_latent=inverted_latent,
        order=args.order,
        save_intermediates=True,
        feature_path=args.feature_path,
        inject_step=args.inject
    )
    
    # Prepare data in the format expected by the original analysis functions
    diffusion_data = {
        'inversion_intermediate_latents': inversion_data['intermediate_latents'],
        'inversion_intermediate_scores': inversion_data['intermediate_scores'],
        'forward_intermediate_latents': generation_result.intermediate_latents,
        'forward_intermediate_scores': generation_result.intermediate_scores,
        'final_latent': generation_result.final_latent,
        'inverted_latent': inverted_latent.cpu(),
        'timesteps': inversion_data['timesteps'],
        'opts': {
            'source_prompt': args.source_prompt,
            'target_prompt': args.target_prompt,
            'width': width,
            'height': height,
            'num_steps': args.num_steps,
            'guidance': args.guidance,
            'seed': None
        },
        'init_image_path': args.source_img_dir,
        'guidance': args.guidance,
        'order': args.order,
        'model_name': args.name
    }
    
    # Save the combined data
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    pt_file_path = f"{args.output_dir}/diffusion_data_{args.order}.pt"
    torch.save(diffusion_data, pt_file_path)
    print(f"Diffusion data saved to: {pt_file_path}")
    
    # Save the final generated image
    output_image_path = f"{args.output_dir}/generated_{args.order}.png"
    generation_result.final_image.save(output_image_path, quality=95, subsampling=0)
    print(f"Generated image saved to: {output_image_path}")
    
    # Also save using the manager's built-in method
    manager.save_generation_result(generation_result, args.output_dir, f"full_results_{args.order}")
    
    return pt_file_path


def main(args):
    """Main function that uses FluxModelManager."""
    
    if args.run_mode == 'inversion':
        # Use the new manager-based approach
        pt_file_path = run_inversion_with_manager(args)
        
    else:
        # For analysis modes, use the existing functions
        pt_file_path = f"{args.output_dir}/diffusion_data_{args.order}.pt"
        
        if args.run_mode == 'single_analysis':
            analyze_single_step_denoising(pt_file_path, "cuda" if torch.cuda.is_available() else "cpu")
        elif args.run_mode == 'multi_analysis':
            analyze_multi_step_denoising(pt_file_path, "cuda" if torch.cuda.is_available() else "cpu")
        elif args.run_mode == 'compare':
            compare_denoising_strategies(pt_file_path, "cuda" if torch.cuda.is_available() else "cpu")
        else:
            raise ValueError(f"Invalid run mode: {args.run_mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RF-Edit with FluxModelManager')
    
    parser.add_argument('--name', default='flux-dev', type=str,
                        help='flux model')
    parser.add_argument('--source_img_dir', default='', type=str,
                        help='The path of the source image')
    parser.add_argument('--source_prompt', type=str,
                        help='describe the content of the source image (or leaves it as null)')
    parser.add_argument('--target_prompt', type=str,
                        help='describe the requirement of editing')
    parser.add_argument('--feature_path', type=str, default='feature',
                        help='the path to save the feature')
    parser.add_argument('--guidance', type=float, default=5,
                        help='guidance scale')
    parser.add_argument('--num_steps', type=int, default=50,
                        help='the number of timesteps for inversion and denoising')
    parser.add_argument('--inject', type=int, default=20,
                        help='the number of timesteps which apply the feature sharing')
    parser.add_argument('--output_dir', default='output', type=str,
                        help='the path of the edited image')
    parser.add_argument('--order', type=int, default=2,
                        help='the order of the diffusion model')
    parser.add_argument('--offload', action='store_true', 
                        help='set it to True if the memory of GPU is not enough')
    parser.add_argument('--run_mode', type=str, default='inversion', 
                        choices=['inversion', 'single_analysis', 'multi_analysis', 'compare'],
                        help='the mode of the program')
    
    args = parser.parse_args()
    
    main(args) 