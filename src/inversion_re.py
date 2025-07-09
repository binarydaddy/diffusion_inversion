import os
import re
import time
from dataclasses import dataclass
from glob import iglob
import argparse
import torch
from einops import rearrange
from PIL import ExifTags, Image
import math

from flux.sampling import denoise, get_schedule, prepare, unpack, denoise_single_step_to_x0, denoise_starting_particular_step
from flux.util import (configs, embed_watermark, load_ae, load_clip,
                       load_flow_model, load_t5)
from transformers import pipeline
from PIL import Image
import numpy as np
import json
from utils import plot_latent_spectrogram, equalize_frequency_magnitudes, get_latent_statistics
import os

NSFW_THRESHOLD = 0.85



@dataclass
class SamplingOptions:
    source_prompt: str
    target_prompt: str
    # prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None

@torch.inference_mode()
def encode(init_image, torch_device, ae):
    init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
    init_image = init_image.unsqueeze(0) 
    init_image = init_image.to(torch_device)
    init_image = ae.encode(init_image.to()).to(torch.bfloat16)
    return init_image

def callback(log):
    t = log["t"]
    latent = log["latent"]
    score = log["score"]
    print(f"t: {t}")
    print(f"latent: {latent.shape}")
    print(f"score: {score.shape}")
    return t, latent, score

@torch.inference_mode()
def latent_to_image(latent, opts, ae):
    batch_latent = unpack(latent.float(), opts.width, opts.height)
    for latent in batch_latent:
        latent = latent.unsqueeze(0)
        latent = ae.decode(latent)
        latent = latent.clamp(-1, 1)
        latent = rearrange(latent[0], "c h w -> h w c")
        latent = Image.fromarray((127.5 * (latent + 1.0)).cpu().byte().numpy())
    return latent


def initialize_models(name="flux-dev", device="cuda", offload=False):
    """
    Initialize all required models for diffusion and analysis.
    
    Args:
        name (str): Model name (e.g., "flux-dev", "flux-schnell")
        device (str): Device to load models on
        offload (bool): Whether to offload models to CPU to save memory
    
    Returns:
        dict: Dictionary containing all initialized models
    """
    from flux.util import load_t5, load_clip, load_flow_model, load_ae, configs
    
    if name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Got unknown model name: {name}, chose from {available}")
    
    torch_device = torch.device(device)
    
    print(f"Initializing models: {name}")
    
    # Initialize text encoders
    print("Loading T5 text encoder...")
    t5 = load_t5(torch_device, max_length=256 if name == "flux-schnell" else 512)
    
    print("Loading CLIP text encoder...")
    clip = load_clip(torch_device)
    
    # Initialize main diffusion model
    print("Loading diffusion model...")
    model = load_flow_model(name, device="cpu" if offload else torch_device)
    
    # Initialize autoencoder
    print("Loading autoencoder...")
    ae = load_ae(name, device="cpu" if offload else torch_device)
    
    # Handle offloading
    if offload:
        print("Offloading models to CPU...")
        model.cpu()
        torch.cuda.empty_cache()
        ae.encoder.to(torch_device)
    
    models = {
        't5': t5,
        'clip': clip,
        'model': model,
        'ae': ae,
        'torch_device': torch_device,
        'name': name,
        'offload': offload
    }
    
    print("Model initialization complete.")
    return models


def compute_psnr(img1, img2):
    # Convert PIL images to numpy arrays
    if isinstance(img1, Image.Image):
        img1 = np.array(img1)
    if isinstance(img2, Image.Image):
        img2 = np.array(img2)
    
    # Ensure same dimensions
    if img1.shape != img2.shape:
        # Resize img2 to match img1
        img2_pil = Image.fromarray(img2)
        img2_pil = img2_pil.resize((img1.shape[1], img1.shape[0]), Image.LANCZOS)
        img2 = np.array(img2_pil)
    
    # Convert to float and normalize to [0, 1]
    img1 = img1.astype(np.float64) / 255.0
    img2 = img2.astype(np.float64) / 255.0
    
    # Compute MSE
    mse = np.mean((img1 - img2) ** 2)
    
    # Avoid division by zero
    if mse == 0:
        return float('inf')
    
    # Compute PSNR
    psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    return psnr

def plot_psnr_scatter(psnr_results, output_dir):
    """
    Create a scatter plot of PSNR values vs timesteps.
    
    Args:
        psnr_results (dict): Dictionary with timestep keys and PSNR values
        output_dir (str): Directory to save the plot
    """
    import matplotlib.pyplot as plt
    
    # Convert keys to float for proper numeric x-axis
    x_values = [float(k) for k in psnr_results.keys()]
    y_values = list(psnr_results.values())
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, y_values, alpha=0.7, s=50, color='blue', edgecolors='black', linewidth=0.5)
    plt.xlabel('Timestep')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR vs Timestep')
    plt.grid(True, alpha=0.3)
    
    # Add some styling
    plt.tight_layout()
    
    # Save the plot
    plot_path = f"{output_dir}/psnr_scatter_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"PSNR scatter plot saved to: {plot_path}")
    return plot_path

def plot_l2_scatter(l2_results, output_dir, plotname="l2_scatter_plot"):
    """
    Create a scatter plot of L2 values vs timesteps.
    
    Args:
        l2_results (dict): Dictionary with timestep keys and L2 values
        output_dir (str): Directory to save the plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert keys to float for proper numeric x-axis
    x_values = [float(k) for k in l2_results.keys()]
    y_values = list(l2_results.values())
    
    # Sort by timestep for better visualization
    sorted_data = sorted(zip(x_values, y_values))
    x_values, y_values = zip(*sorted_data)
    
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    plt.scatter(x_values, y_values, alpha=0.8, s=60, color='blue', 
                edgecolors='darkblue', linewidth=1, label='L2 Difference')
    
    # Add trend line
    plt.plot(x_values, y_values, alpha=0.5, color='red', linewidth=1, 
             linestyle='--', label='Trend Line')
    
    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('L2 Difference', fontsize=12)
    plt.title('L2 Difference vs Timestep', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add annotations for min/max points if there are enough data points
    if len(y_values) > 1:
        min_idx = np.argmin(y_values)
        max_idx = np.argmax(y_values)
        
        plt.annotate(f'Min: {y_values[min_idx]:.4f}', 
                     xy=(x_values[min_idx], y_values[min_idx]),
                     xytext=(10, 10), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.annotate(f'Max: {y_values[max_idx]:.4f}', 
                     xy=(x_values[max_idx], y_values[max_idx]),
                     xytext=(10, 10), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Add summary statistics as text box
    if y_values:
        stats_text = f'Statistics:\n'
        stats_text += f'Points: {len(y_values)}\n'
        stats_text += f'Mean: {np.mean(y_values):.4f}\n'
        stats_text += f'Std: {np.std(y_values):.4f}\n'
        stats_text += f'Min: {np.min(y_values):.4f}\n'
        stats_text += f'Max: {np.max(y_values):.4f}'
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add some styling
    plt.tight_layout()
    
    # Save the plot
    plot_path = f"{output_dir}/{plotname}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"L2 scatter plot saved to: {plot_path}")
    
    # Print summary statistics to console
    if y_values:
        print(f"L2 Difference Summary:")
        print(f"  Number of points: {len(y_values)}")
        print(f"  Mean: {np.mean(y_values):.6f}")
        print(f"  Std: {np.std(y_values):.6f}")
        print(f"  Min: {np.min(y_values):.6f}")
        print(f"  Max: {np.max(y_values):.6f}")
    
    return plot_path

def plot_log_likelihood_scatter(likelihood_results, output_dir):
    """
    Create a scatter plot of log likelihood values vs timesteps.
    
    Args:
        likelihood_results (dict): Dictionary with timestep keys and log likelihood values
        output_dir (str): Directory to save the plot
    """
    import matplotlib.pyplot as plt
    
    # Convert keys to float for proper numeric x-axis
    x_values = [float(k) for k in likelihood_results.keys()]
    y_values = list(likelihood_results.values())

    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, y_values, alpha=0.7, s=50, color='blue', edgecolors='black', linewidth=0.5)
    plt.xlabel('Timestep')
    plt.ylabel('log likelihood')
    plt.title('Log Likelihood vs Timestep')
    plt.grid(True, alpha=0.3)
    
    # Add some styling
    plt.tight_layout()
    
    # Save the plot
    plot_path = f"{output_dir}/loglikelihood_scatter_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Log likelihood scatter plot saved to: {plot_path}")
    return plot_path


def render_latents_and_compute_psnr_and_compute_likelihood(processed_latents, init_image, analysis_dir, opts, ae, device):
    """
    Analyze processed latents by generating images, computing PSNR, and creating plots.
    
    Args:
        processed_latents (dict): Dictionary of timestep -> latent tensor
        init_image (np.array): Original image for PSNR comparison
        analysis_dir (str): Directory to save analysis results
        opts (SamplingOptions): Sampling options object
        ae: Autoencoder model for decoding latents
        
    Returns:
        dict: PSNR results dictionary
    """
    # Ensure analysis directory exists
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
    
    # Compute PSNR and save images
    psnr_results = {}
    likelihood_results = {}

    for k, latent in processed_latents.items():
        print(f"Generating image for timestep {k}")
        
        # Move latent to device and generate image
        latent = latent.to(device)
        img = latent_to_image(latent, opts, ae)
        
        # Save image
        img.save(f"{analysis_dir}/latent_step_{k:02f}.png")
        
        # Compute PSNR
        psnr_value = compute_psnr(img, init_image)
        psnr_results[f"{k:02f}"] = psnr_value
        print(f"PSNR for step {k:02f}: {psnr_value:.2f} dB")

        # Compute likelihood under standard normal distribution
        likelihood_value = multivariate_gaussian_log_likelihood(latent.reshape(1, -1))
        likelihood_results[f"{k:02f}"] = likelihood_value.item()
        print(f"Likelihood for step {k:02f}: {likelihood_value.item()}")
    
    # Save PSNR results to JSON
    psnr_file_path = f"{analysis_dir}/psnr_results.json"
    with open(psnr_file_path, 'w') as f:
        json.dump(psnr_results, f, indent=2)
    print(f"PSNR results saved to: {psnr_file_path}")

    # Save likelihood results to JSON
    likelihood_file_path = f"{analysis_dir}/likelihood_results.json"
    with open(likelihood_file_path, 'w') as f:
        json.dump(likelihood_results, f, indent=2)
    print(f"Likelihood results saved to: {likelihood_file_path}")

    # Create scatter plot
    plot_psnr_scatter(psnr_results, analysis_dir)
    plot_log_likelihood_scatter(likelihood_results, analysis_dir)
    
    print(f"Analysis complete. Results saved in: {analysis_dir}")
    return psnr_results


def analyze_diffusion_data(pt_file_path, device="cuda"):
    """
    Load diffusion data from .pt file and perform analysis including PSNR computation,
    image generation, and plotting.
    
    Args:
        pt_file_path (str): Path to the .pt file containing diffusion data
        device (str): Device to use for processing
    """
    print(f"Loading diffusion data from: {pt_file_path}")
    
    # Load the data
    data = torch.load(pt_file_path, map_location='cpu')
    
    # Extract data
    inversion_latents = data['inversion_latents']
    forward_latents = data['forward_latents']
    opts_dict = data['opts']
    init_image_path = data['init_image_path']
    order = data['order']
    
    # Reconstruct opts object
    opts = SamplingOptions(
        source_prompt=opts_dict['source_prompt'],
        target_prompt=opts_dict['target_prompt'],
        width=opts_dict['width'],
        height=opts_dict['height'],
        num_steps=opts_dict['num_steps'],
        guidance=opts_dict['guidance'],
        seed=opts_dict['seed']
    )
    
    # Load original image for PSNR comparison
    init_image = np.array(Image.open(init_image_path).convert('RGB'))
    
    # Setup output directory
    output_dir = os.path.dirname(pt_file_path)
    analysis_dir = f"{output_dir}/analysis_{order}"
    
    # Get model name from saved data, fallback to flux-dev
    model_name = data.get('model_name', 'flux-dev')
    
    # Initialize all models using the saved model name
    models = initialize_models(name=model_name, device=device, offload=False)
    ae = models['ae']
    
    print(f"Processing {len(inversion_latents)} inversion steps...")
    
    # Process denoising for selected timesteps (same logic as before)
    processed_latents = {}
    for i, k in enumerate(sorted(inversion_latents.keys())):
        if i % 5 == 0 or i == len(inversion_latents.keys()) - 1 or i >= len(inversion_latents.keys()) - 5:
            print(f"Processing timestep {k}")
            if k in forward_latents:
                processed_latents[k] = forward_latents[k]
    
    # Analyze the processed latents
    psnr_results = render_latents_and_compute_psnr_and_compute_likelihood(
        processed_latents=processed_latents,
        init_image=init_image,
        analysis_dir=analysis_dir,
        opts=opts,
        ae=ae,
        device=device
    )
    
    return psnr_results


def analyze_single_step_denoising(pt_file_path, device="cuda"):
    """
    Analyze using single-step denoising from intermediate latents.
    
    Args:
        pt_file_path (str): Path to the .pt file containing diffusion data
        device (str): Device to use for processing
        
    Returns:
        dict: PSNR results from single-step denoising
    """
    print(f"Loading diffusion data for single-step analysis: {pt_file_path}")
    
    # Load the data
    data = torch.load(pt_file_path, map_location='cpu')
    
    # Extract data
    inversion_latents = data['inversion_latents']
    opts_dict = data['opts']
    init_image_path = data['init_image_path']
    order = data['order']
    
    # Reconstruct opts object
    opts = SamplingOptions(
        source_prompt=opts_dict['source_prompt'],
        target_prompt=opts_dict['target_prompt'],
        width=opts_dict['width'],
        height=opts_dict['height'],
        num_steps=opts_dict['num_steps'],
        guidance=opts_dict['guidance'],
        seed=opts_dict['seed']
    )
    
    # Load original image for PSNR comparison
    init_image_pil = np.array(Image.open(init_image_path).convert('RGB'))
    
    # Setup output directory
    output_dir = os.path.dirname(pt_file_path)
    analysis_dir = f"{output_dir}/single_step_analysis_{order}"
    
    # Initialize all models (need t5, clip, model, ae)
    model_name = data.get('model_name', 'flux-dev')
    models = initialize_models(name=model_name, device=device, offload=False)
    ae = models['ae']
    model = models['model']
    t5 = models['t5']
    clip = models['clip']
    torch_device = models['torch_device']
    timesteps = data['timesteps']
    
    # Encode the initial image for processing
    init_image_encoded = encode(init_image_pil, torch_device, ae)
    
    # Create inp dictionary using prepare function
    inp = prepare(t5, clip, init_image_encoded, prompt=opts.source_prompt)
    
    # Create info dictionary
    info = {}
    info['feature_path'] = 'feature'  # Default feature path
    info['feature'] = {}
    info['inject_step'] = 0  # Default inject step
    
    print("Performing single-step denoising analysis...")
    
    # Perform single-step denoising for selected timesteps
    processed_latents = {}
    for i, k in enumerate(sorted(inversion_latents.keys())):
        if i % 5 == 0 or i == len(inversion_latents.keys()) - 1 or i >= len(inversion_latents.keys()) - 5:
            print(f"Single-step denoising for timestep {k}")
            inp_with_noisy_latent = inp.copy()
            inp_with_noisy_latent["img"] = inversion_latents[k].to(device)
            z, _ = denoise_single_step_to_x0(model, **inp_with_noisy_latent, target_t=k, guidance=opts.guidance, inverse=False, info=info, order=order)
            processed_latents[k] = z
    
    # Analyze the processed latents
    psnr_results =  render_latents_and_compute_psnr_and_compute_likelihood(
        processed_latents=processed_latents,
        init_image=init_image_pil,
        analysis_dir=analysis_dir,
        opts=opts,
        ae=ae,
        device=device
    )
    
    return psnr_results


def analyze_multi_step_denoising(pt_file_path, device="cuda"):
    """
    Analyze using multi-step denoising (original analyze_diffusion_data logic).
    
    Args:
        pt_file_path (str): Path to the .pt file containing diffusion data
        device (str): Device to use for processing
        
    Returns:
        dict: PSNR results from multi-step denoising
    """
    print(f"Loading diffusion data for multi-step analysis: {pt_file_path}")
    
    # Load the data
    data = torch.load(pt_file_path, map_location='cpu')
    
    # Extract data
    inversion_latents = data['inversion_latents']
    forward_latents = data['forward_latents']
    opts_dict = data['opts']
    init_image_path = data['init_image_path']
    order = data['order']
    
    # Reconstruct opts object
    opts = SamplingOptions(
        source_prompt=opts_dict['source_prompt'],
        target_prompt=opts_dict['target_prompt'],
        width=opts_dict['width'],
        height=opts_dict['height'],
        num_steps=opts_dict['num_steps'],
        guidance=opts_dict['guidance'],
        seed=opts_dict['seed']
    )
    
    # Load original image for PSNR comparison
    init_image_pil = np.array(Image.open(init_image_path).convert('RGB'))
    
    # Setup output directory
    output_dir = os.path.dirname(pt_file_path)
    analysis_dir = f"{output_dir}/multi_step_analysis_{order}"
    
    # Initialize all models (need t5, clip, model, ae)
    model_name = data.get('model_name', 'flux-dev')
    models = initialize_models(name=model_name, device=device, offload=False)
    ae = models['ae']
    model = models['model']
    t5 = models['t5']
    clip = models['clip']
    torch_device = models['torch_device']
    timesteps = data['timesteps']
    
    # Encode the initial image for processing
    init_image_encoded = encode(init_image_pil, torch_device, ae)
    
    # Create inp dictionary using prepare function with target prompt for multi-step
    inp = prepare(t5, clip, init_image_encoded, prompt=opts.target_prompt)
    
    # Create info dictionary
    info = {}
    info['feature_path'] = 'feature'  # Default feature path
    info['feature'] = {}
    info['inject_step'] = 0  # Default inject step
    
    print("Performing multi-step denoising analysis...")
    
    print(f"timesteps: {timesteps}")

    # Perform multi-step denoising for selected timesteps
    processed_latents = {}
    for i, k in enumerate(sorted(inversion_latents.keys())):
        if i % 5 == 0 or i == len(inversion_latents.keys()) - 1 or i >= len(inversion_latents.keys()) - 5:
            if k in forward_latents:
                print(f"Multi-step denoising for timestep {k}")
                # Start from the inverted latent at timestep k, not the clean image
                inp_with_noisy_latent = inp.copy()
                inp_with_noisy_latent["img"] = inversion_latents[k].to(device)
                z, _ = denoise_starting_particular_step(model, **inp_with_noisy_latent, timesteps=timesteps, target_t=k, guidance=opts.guidance, inverse=False, info=info, order=order)
                processed_latents[k] = z
    
    # Analyze the processed latents
    psnr_results = render_latents_and_compute_psnr_and_compute_likelihood(
        processed_latents=processed_latents,
        init_image=init_image_pil,
        analysis_dir=analysis_dir,
        opts=opts,
        ae=ae,
        device=device
    )
    
    return psnr_results

@torch.inference_mode()
def run_model(device, name="flux-dev", source_prompt="", target_prompt="", guidance=3.5, output_dir="", order=2, num_steps=25, offload=False, latent=None, models=None, width=1024, height=1024, seed=None, feature_path="feature", inject_step=0, output_filename="generated_from_latent.png"):
    """
    Run forward denoising starting from a given latent.
    
    Args:
        latent: Input latent tensor to start denoising from
        device: Device to run on
        name: Model name (flux-dev or flux-schnell)
        source_prompt: Source prompt (not used in forward pass but needed for preparation)
        target_prompt: Target prompt for denoising
        guidance: Guidance scale
        output_dir: Output directory
        order: Solver order (1 or 2)
        num_steps: Number of denoising steps
        offload: Whether to offload models
        seed: Random seed
        feature_path: Feature path for injection
        inject_step: Injection step
    
    Returns:
        Generated image and final latent
    """
    torch.set_grad_enabled(False)
    
    if num_steps is None:
        num_steps = 4 if name == "flux-schnell" else 25

    # Initialize all models
    if models is None:
        models = initialize_models(name=name, device=device, offload=offload)
    t5 = models['t5']
    clip = models['clip']
    model = models['model']
    ae = models['ae']
    torch_device = models['torch_device']
    
    # Create sampling options
    opts = SamplingOptions(
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        width=width,  # Convert latent dimensions to pixel dimensions
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed if seed is not None else torch.randint(0, 2**32-1, (1,)).item(),
    )

    print(f"Starting denoising with latent shape: {latent.shape}")
    print(f"Target prompt: {target_prompt}")
    print(f"Guidance: {guidance}, Steps: {num_steps}")

    # Setup models for text encoding
        # Prepare text embeddings using the target prompt
    # We use a dummy image tensor with the right shape for prepare()
    init_image = None
    init_image = np.array(Image.open(image).convert('RGB'))
    
    shape = init_image.shape

    new_h = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
    new_w = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16

    init_image = init_image[:new_h, :new_w, :]

    width, height = init_image.shape[0], init_image.shape[1]

    if offload:
        t5, clip = t5.cpu(), clip.cpu()
        torch.cuda.empty_cache()
        ae = ae.cuda()

    init_image = encode(init_image, torch_device, ae)

    if latent is not None:
        latent = latent.to(device)
    else:
        latent = torch.randn_like(init_image)
    
    if offload:
        ae = ae.cpu()
        torch.cuda.empty_cache()
        t5, clip = t5.to(torch_device), clip.to(torch_device)

    # Setup info dictionary
    info = {}
    info['feature_path'] = feature_path
    info['feature'] = {}
    info['inject_step'] = inject_step
    
    # Create feature directory if it doesn't exist
    if not os.path.exists(feature_path):
        os.makedirs(feature_path, exist_ok=True)

    inp = prepare(t5, clip, init_image, prompt=target_prompt)
    
    # Replace the dummy image with our actual latent
    inp["img"] = latent
    
    # Get timestep schedule
    timesteps = get_schedule(opts.num_steps, latent.shape[1], shift=(name != "flux-schnell"))
    
    print(f"Timesteps: {timesteps}")
    print(f"Number of timesteps: {len(timesteps)}")

    # Offload text encoders, load diffusion model
    if offload:
        t5, clip = t5.cpu(), clip.cpu()
        torch.cuda.empty_cache()
        model = model.to(torch_device)

    # Setup callback to collect intermediate results
    intermediate_latents = {}
    intermediate_scores = {}
    
    def callback(log):
        t = log["t"]
        intermediate_latents[t] = log["latent"].cpu()
        intermediate_scores[t] = log["score"].cpu()

    # Run forward denoising
    print("Starting forward denoising...")
    final_latent, final_info = denoise(
        model, 
        **inp, 
        timesteps=timesteps, 
        guidance=guidance, 
        inverse=False, 
        info=info, 
        order=order, 
        callback=callback
    )

    # Decode final latent to image
    if offload:
        model.cpu()
        torch.cuda.empty_cache()
        ae.decoder.to(final_latent.device)

    # Decode latent to pixel space
    batch_x = unpack(final_latent.float(), opts.width, opts.height)
    
    with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
        x = ae.decode(batch_x)

    # Convert to PIL image
    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")
    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
    
    # Save the generated image
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    output_path = f"{output_dir}/{output_filename}"
    img.save(output_path)
    print(f"Generated image saved to: {output_path}")
    
    # Clean up
    if offload:
        ae.decoder.cpu()
        torch.cuda.empty_cache()

    return {
        'image': img,
        'final_latent': final_latent,
        'intermediate_latents': intermediate_latents,
        'intermediate_scores': intermediate_scores,
        'output_path': output_path,
        'opts': opts,
        'timesteps': timesteps,
        'inputs': inp,
    }    




def compare_denoising_strategies(pt_file_path, device="cuda"):
    """
    Compare different denoising strategies on the same diffusion data.
    
    Args:
        pt_file_path (str): Path to the .pt file containing diffusion data
        device (str): Device to use for processing
        
    Returns:
        dict: Dictionary containing PSNR results from different strategies
    """
    print(f"Comparing denoising strategies for: {pt_file_path}")
    
    # Run single-step analysis
    print("\n" + "="*50)
    print("SINGLE-STEP DENOISING ANALYSIS")
    print("="*50)
    single_step_results = analyze_single_step_denoising(pt_file_path, device)
    
    # Run multi-step analysis
    print("\n" + "="*50)
    print("MULTI-STEP DENOISING ANALYSIS")
    print("="*50)
    multi_step_results = analyze_multi_step_denoising(pt_file_path, device)
    
    # Compare results
    print("\n" + "="*50)
    print("COMPARISON SUMMARY")
    print("="*50)
    
    comparison_results = {
        'single_step': single_step_results,
        'multi_step': multi_step_results
    }
    
    # Calculate average PSNR for each method
    if single_step_results:
        avg_single = sum(single_step_results.values()) / len(single_step_results)
        print(f"Single-step average PSNR: {avg_single:.2f} dB")
    
    if multi_step_results:
        avg_multi = sum(multi_step_results.values()) / len(multi_step_results)
        print(f"Multi-step average PSNR: {avg_multi:.2f} dB")
    
    # Save comparison results
    output_dir = os.path.dirname(pt_file_path)
    comparison_file = f"{output_dir}/denoising_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    print(f"Comparison results saved to: {comparison_file}")
    
    return comparison_results

def run_and_collect_inversion_data(args, 
                                   seed: int | None = None, 
                                   device: str = "cuda" if torch.cuda.is_available() else "cpu", 
                                   num_steps: int | None = None, 
                                   loop: bool = False, 
                                   offload: bool = False, 
                                   order: int=2, 
                                   add_sampling_metadata: bool = True,
                                   fastforward_steps = None):
    """
        Sample the flux model. Either interactively (set `--loop`) or run for a
        single image.

        Args:
            name: Name of the model to load
            height: height of the sample in pixels (should be a multiple of 16)
            width: width of the sample in pixels (should be a multiple of 16)
            seed: Set a seed for sampling
            output_name: where to save the output image, `{idx}` will be replaced
                by the index of the sample
            prompt: Prompt used for sampling
            device: Pytorch device
            num_steps: number of sampling steps (default 4 for schnell, 50 for guidance distilled)
            loop: start an interactive session and sample multiple times
            guidance: guidance value used for guidance distillation
            add_sampling_metadata: Add the prompt to the image Exif metadata
    """
    
    torch.set_grad_enabled(False)
    name = args.name
    source_prompt = args.source_prompt
    target_prompt = args.target_prompt
    guidance = args.guidance
    output_dir = args.output_dir
    order = args.order
    num_steps = args.num_steps
    offload = args.offload

    nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)

    if num_steps is None:
        num_steps = 4 if name == "flux-schnell" else 25

    # Initialize all models
    models = initialize_models(name=name, device=device, offload=offload)
    t5 = models['t5']
    clip = models['clip']
    model = models['model']
    ae = models['ae']
    torch_device = models['torch_device']
    
    init_image = None
    init_image = np.array(Image.open(args.source_img_dir).convert('RGB'))
    
    shape = init_image.shape

    new_h = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
    new_w = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16

    init_image = init_image[:new_h, :new_w, :]

    width, height = init_image.shape[0], init_image.shape[1]
    init_image = encode(init_image, torch_device, ae)

    rng = torch.Generator(device="cpu")
    opts = SamplingOptions(
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
    )

    if loop:
        opts = parse_prompt(opts)

    while opts is not None:
        if opts.seed is None:
            opts.seed = rng.seed()
        print(f"Generating with seed {opts.seed}:\n{opts.source_prompt}")
        t0 = time.perf_counter()

        opts.seed = None
        if offload:
            ae = ae.cpu()
            torch.cuda.empty_cache()
            t5, clip = t5.to(torch_device), clip.to(torch_device)

        info = {}
        info['feature_path'] = args.feature_path
        info['feature'] = {}
        info['inject_step'] = args.inject
        if not os.path.exists(args.feature_path):
            os.mkdir(args.feature_path)

        inp = prepare(t5, clip, init_image, prompt=opts.source_prompt)
        inp_target = prepare(t5, clip, init_image, prompt=opts.target_prompt)
        timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))

        print(f"timesteps: {timesteps}")
        print(f"timesteps_len: {len(timesteps)}")

        # offload TEs to CPU, load model to gpu
        if offload:
            t5, clip = t5.cpu(), clip.cpu()
            torch.cuda.empty_cache()
            model = model.to(torch_device)

        # Setup callback to collect intermediate results
        intermediate_latents = {}
        intermediate_scores = {}
        
        def callback(log):
            t = log["t"]
            intermediate_latents[t] = log["latent"].cpu()  # Move to CPU to save memory
            intermediate_scores[t] = log["score"].cpu()

        # inversion initial noise        
        z, info = denoise(model, **inp, timesteps=timesteps, guidance=guidance, inverse=True, info=info, order=order, callback=callback, fastforward_steps=fastforward_steps)
        
        inp_target["img"] = z

        timesteps = get_schedule(opts.num_steps, inp_target["img"].shape[1], shift=(name != "flux-schnell"))

        # denoise initial noise  
        forward_intermediate_latents = {}
        forward_intermediate_scores = {}
        
        def forward_callback(log):
            t = log["t"]
            forward_intermediate_latents[t] = log["latent"].cpu()
            forward_intermediate_scores[t] = log["score"].cpu()
            
        x, _ = denoise(model, **inp_target, timesteps=timesteps, guidance=guidance, inverse=False, info=info, order=order, callback=forward_callback)

        # Save all diffusion data for later analysis
        diffusion_data = {
            'inversion_latents': intermediate_latents,
            'inversion_scores': intermediate_scores,
            'forward_latents': forward_intermediate_latents,
            'forward_scores': forward_intermediate_scores,
            'final_latent': x.cpu(),
            'inverted_latent': z.cpu(),
            'timesteps': timesteps,
            'opts': {
                'source_prompt': opts.source_prompt,
                'target_prompt': opts.target_prompt,
                'width': opts.width,
                'height': opts.height,
                'num_steps': opts.num_steps,
                'guidance': opts.guidance,
                'seed': opts.seed
            },
            'init_image_path': args.source_img_dir,
            'guidance': guidance,
            'order': order,
            'model_name': name  # Save model name for analysis
        }
        
        # Save to .pt file
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        pt_file_path = f"{output_dir}/diffusion_data_{order}.pt"
        torch.save(diffusion_data, pt_file_path)
        print(f"Diffusion data saved to: {pt_file_path}")

        if offload:
            model.cpu()
            torch.cuda.empty_cache()
            ae.decoder.to(x.device)

        # decode latents to pixel space
        batch_x = unpack(x.float(), opts.width, opts.height)
        batch_z = unpack(z.float(), opts.width, opts.height)

        for x, z in zip(batch_x, batch_z):
            x = x.unsqueeze(0)
            z = z.unsqueeze(0)
            output_name = os.path.join(output_dir, "img_{idx}.jpg")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                idx = 0
            else:
                fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
                if len(fns) > 0:
                    idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
                else:
                    idx = 0

            with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                x = ae.decode(x)
                z = ae.decode(z)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            output_name = f"{output_dir}/image_{idx}_{order}.png"

            print(f"Done in {t1 - t0:.1f}s. Saving {output_name}")
            # bring into PIL format and save
            x = x.clamp(-1, 1)
            x = rearrange(x[0], "c h w -> h w c")

            z = z.clamp(-1, 1)
            z = rearrange(z[0], "c h w -> h w c")
            z = Image.fromarray((127.5 * (z + 1.0)).cpu().byte().numpy())
            z.save(f"{output_dir}/latents_{idx}_{order}.png")

            img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
            nsfw_score = [x["score"] for x in nsfw_classifier(img) if x["label"] == "nsfw"][0]
            
            if nsfw_score < NSFW_THRESHOLD:
                exif_data = Image.Exif()
                exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
                exif_data[ExifTags.Base.Make] = "Black Forest Labs"
                exif_data[ExifTags.Base.Model] = name
                if add_sampling_metadata:
                    exif_data[ExifTags.Base.ImageDescription] = source_prompt
                img.save(output_name, exif=exif_data, quality=95, subsampling=0)
                idx += 1
            else:
                print("Your generated image may contain NSFW content.")

            if loop:
                print("-" * 80)
                opts = parse_prompt(opts)
            else:
                opts = None    


@torch.inference_mode()
def main(
    args,
    seed: int | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int | None = None,
    loop: bool = False,
    offload: bool = False,
    order: int=2,
    add_sampling_metadata: bool = True,
):
    
    pt_file_path = f"{args.output_dir}/diffusion_data_{args.order}.pt"

    if args.run_mode == 'inversion':
        run_and_collect_inversion_data(args, seed, device, num_steps, loop, offload, order, add_sampling_metadata, fastforward_steps=args.fastforward_steps)
    elif args.run_mode == 'single_analysis':
        analyze_single_step_denoising(pt_file_path, device)
    elif args.run_mode == 'multi_analysis':
        analyze_multi_step_denoising(pt_file_path, device)
    elif args.run_mode == 'forward':
        analyze_diffusion_data(pt_file_path, device)
    else:
        raise ValueError(f"Invalid run mode: {args.run_mode}")
    
def multivariate_gaussian_log_likelihood(x):
    """
    Compute the log-likelihood of x under a multivariate Gaussian with mean=0 and covariance=I.
    This is more numerically stable than computing likelihood directly.
    
    Args:
        x: Tensor of shape (..., d) where d is the dimensionality
        
    Returns:
        log_likelihood: Tensor of shape (...) containing log-likelihood values
    """
    d = x.shape[-1]
    squared_norm = torch.sum(x**2, dim=-1)
    log_normalization = -0.5 * d * math.log(2 * math.pi)
    log_likelihood = log_normalization - 0.5 * squared_norm
    
    return log_likelihood

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='RF-Edit')

    parser.add_argument('--name', default='flux-dev', type=str,
                        help='flux model')
    parser.add_argument('--source_img_dir', default='', type=str,
                        help='The path of the source image')
    parser.add_argument('--source_prompt', type=str,
                        help='describe the content of the source image (or leaves it as null)')
    parser.add_argument('--target_prompt', type=str,
                        help='describe the requirement of editing')
    parser.add_argument('--feature_path', type=str, default='feature',
                        help='the path to save the feature ')
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
    parser.add_argument('--offload', action='store_true', help='set it to True if the memory of GPU is not enough')
    parser.add_argument('--fastforward_steps', type=float, default=None,
                        help='the number of timesteps to fastforward')
    parser.add_argument('--run_mode', type=str, default='inversion', choices=['inversion', 'forward', 'single_analysis', 'multi_analysis'],
                        help='the mode of the program')

    args = parser.parse_args()

    main(args)
