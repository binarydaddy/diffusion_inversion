import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math
import os
from torch import Tensor
from einops import rearrange

def unpack(x: Tensor, height: int, width: int) -> Tensor:
    
    h = math.ceil(height / 16)
    w = math.ceil(width / 16)

    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=h,
        w=w,
        ph=2,
        pw=2,
    )

def pack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b c (h ph) (w pw) -> b (h w) (c ph pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
    )

def get_timestep_list_from_intermediate_latents(intermediate_latents) -> list:
    """
        This function returns the timestep list from the intermediate latents.
        
        Args:
            Input: intermediate_latents
    """
    return sorted(list(intermediate_latents.keys()))

def get_aligned_generation_intermediate_latents(generation_run_data, width=1024, height=1024) -> dict:
    """
        This function aligns the generation run data so that intermediate_latents[1.0] is initial noise, and rest is shifted to lower timestep by 1 index.
        
        Current generation run data is always shifted by 1 index timestep.
        e.g. intermediate_latents[1.0] is the result of denoising step at 1.0.

        Args:
            Input: generation_run_data
            Output: dict of intermediate latents with timesteps as keys.
    """

    # These are latents obtained from diffusion denoising process.
    intermediate_latents_timesteps = sorted(list(generation_run_data['intermediate_latents'].keys()), reverse=True)
    intermediate_latents = generation_run_data['intermediate_latents']

    gt_latents = {}
    for i, k in enumerate(intermediate_latents_timesteps):
        
        if i == len(intermediate_latents_timesteps) - 1:
            continue
        
        v = intermediate_latents[k]
        next_ts = intermediate_latents_timesteps[i+1]
        gt_latents[next_ts] = v

    initial_latent = generation_run_data['initial_latent']
    if len(initial_latent.shape) == 4:
        initial_latent = pack(initial_latent, width, height)

    gt_latents[1.0] = initial_latent

    return gt_latents

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
    log_likelihood = log_likelihood.to(torch.float32)
    
    return log_likelihood

def normalize_latent(latent):
    original_dtype = latent.dtype
    latent = latent.to(torch.float32)
    latent_shape = latent.shape

    latent_flatten = latent.reshape(-1)
            
            # Normalize to mean=0, std=1
    latent_mean = latent_flatten.mean()
    latent_std = latent_flatten.std()
    latent_normalized = (latent_flatten - latent_mean) / latent_std
            
            # Reshape back to original shape
    latent_norm = latent_normalized.reshape(latent_shape)    

    return latent_norm.to(original_dtype)

def plot_latent_spectrogram(latent, output_path, title="Latent Frequency Spectrum"):
    """
    Plot and save the frequency domain representation (spectrogram) of a latent tensor.
    
    Args:
        latent (torch.Tensor): Input latent tensor
        output_path (str): Path to save the plot
        title (str): Title for the plot
    """
    
    latent = latent.to(torch.float32)
    
    # Handle different tensor dimensions
    if latent.dim() == 4:  # (batch, channels, height, width)
        # Take first batch and average across channels
        latent_2d = latent[0].mean(dim=0)
    elif latent.dim() == 3:  # (channels, height, width)
        # Average across channels
        latent_2d = latent.mean(dim=0)
    elif latent.dim() == 2:  # (height, width)
        latent_2d = latent
    else:
        raise ValueError(f"Unsupported tensor dimension: {latent.dim()}")
    
    # Compute 2D FFT and shift zero frequency to center
    fft_latents = torch.fft.fftshift(torch.fft.fft2(latent_2d))
    
    # Get magnitude spectrum
    magnitudes = torch.abs(fft_latents)
    magnitudes = magnitudes / magnitudes.max()  # Normalize
    magnitudes = magnitudes.pow(0.5)  # Compress dynamic range
    
    # Convert to numpy for plotting
    magnitudes_np = magnitudes.cpu().numpy()
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Use log scale for better visualization
    magnitudes_log = np.log10(magnitudes_np + 1e-10)  # Add small epsilon to avoid log(0)
    
    # Create the spectrogram plot
    im = plt.imshow(magnitudes_log, cmap='viridis', origin='lower', aspect='auto')
    plt.colorbar(im, label='Log Magnitude')
    plt.title(title)
    plt.xlabel('Frequency (kx)')
    plt.ylabel('Frequency (ky)')
    
    # Add frequency axis labels (assuming square input)
    h, w = magnitudes_np.shape
    center_x, center_y = w // 2, h // 2
    
    # Set tick labels to show frequency values relative to center
    x_ticks = np.linspace(0, w-1, 5).astype(int)
    y_ticks = np.linspace(0, h-1, 5).astype(int)
    x_labels = [f"{(x - center_x) / center_x:.1f}" for x in x_ticks]
    y_labels = [f"{(y - center_y) / center_y:.1f}" for y in y_ticks]
    
    plt.xticks(x_ticks, x_labels)
    plt.yticks(y_ticks, y_labels)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Frequency spectrum saved to: {output_path}")
    return magnitudes

def equalize_frequency_magnitudes(latent):
    latent = latent.to(torch.float32)
    # Compute 2D FFT and shift zero frequency component to center for better visualization/processing
    fft_latents = torch.fft.fftshift(torch.fft.fft2(latent))
    
    # Extract magnitude (amplitude) information from complex FFT coefficients
    magnitudes = torch.abs(fft_latents)
    
    # Normalize magnitudes to [0, 1] range by dividing by maximum value
    # magnitudes = magnitudes / magnitudes.max()
    
    # Apply square root to compress dynamic range (reduces contrast in frequency domain)
    magnitudes = magnitudes.pow(0.95)
    
    # Reconstruct complex FFT by combining modified magnitudes with original phase information
    # This preserves spatial structure while equalizing frequency content
    fft_latents = magnitudes * torch.exp(1j * torch.angle(fft_latents))
    
    # Shift frequency components back to original FFT layout (undo fftshift)
    fft_latents = torch.fft.ifftshift(fft_latents)
    
    # Convert back to spatial domain using inverse 2D FFT
    fft_latents = torch.fft.ifft2(fft_latents)
    
    equalized_latents = fft_latents.real.to(torch.bfloat16)
    # Return only real part (imaginary part should be negligible due to symmetry)
    return equalized_latents

def set_frequency_magnitudes_from_gaussian_noise(latent, frequency_ratio=0.5):
    """
    Mix frequencies: low frequencies from original latent + high frequencies from gaussian noise.
    Uses the same pattern as mix_freqs function.
    
    Args:
        latent: Input latent tensor
        frequency_ratio: Ratio of low frequencies to preserve (0.0 to 1.0)
                        0.5 means preserve 50% of frequencies around center
    """
    latent = latent.to(torch.float32)
    gaussian_noise = torch.randn_like(latent)
    
    # Get FFT of both latent and gaussian noise
    fft_latents = torch.fft.fftshift(torch.fft.fft2(latent))
    fft_gaussian = torch.fft.fftshift(torch.fft.fft2(gaussian_noise))

    # Create rectangular frequency mask (like mix_freqs)
    h, w = latent.shape[-2:]
    center_h, center_w = h // 2, w // 2

    mask_h = int(h * frequency_ratio)
    mask_w = int(w * frequency_ratio)
        
    # Create frequency mask for low frequencies
    mask = torch.zeros_like(fft_latents, dtype=torch.bool)
    mask[..., 
         center_h - mask_h//2:center_h + mask_h//2,
         center_w - mask_w//2:center_w + mask_w//2] = True
    
    # Extract low frequencies from original latent
    low_freq_fft = torch.where(mask, fft_latents, torch.zeros_like(fft_latents))

    # Extract high frequencies from gaussian noise
    high_freq_fft = torch.where(~mask, fft_gaussian, torch.zeros_like(fft_gaussian))
    
    # Combine: low frequencies from original + high frequencies from Gaussian noise
    combined_fft = low_freq_fft + high_freq_fft
    
    # Convert back to spatial domain
    combined_latents = torch.fft.ifft2(torch.fft.ifftshift(combined_fft)).real
    
    return combined_latents.to(torch.bfloat16)

def mix_freqs(pipe, latent1, latent2, low_freq_ratio):
    latent1 = pipe._unpack_latents(latent1, 1024, 1024, 8)
    latent2 = pipe._unpack_latents(latent2, 1024, 1024, 8)

    latent1 = latent1.to(torch.float32)
    latent2 = latent2.to(torch.float32)

    fft_latents1 = torch.fft.fftshift(torch.fft.fft2(latent1))
    fft_latents2 = torch.fft.fftshift(torch.fft.fft2(latent2))

    h, w = latent1.shape[-2:]
    center_h, center_w = h // 2, w // 2

    mask_h = int(h * low_freq_ratio)
    mask_w = int(w * low_freq_ratio)
        
    # Create frequency mask
    mask = torch.zeros_like(fft_latents1, dtype=torch.bool)
    mask[..., 
            center_h - mask_h//2:center_h + mask_h//2,
            center_w - mask_w//2:center_w + mask_w//2] = True
    
    # Extract low frequencies from original latents
    low_freq_fft = torch.where(mask, fft_latents1, torch.zeros_like(fft_latents1))

    high_freq_fft = torch.where(~mask, fft_latents2, torch.zeros_like(fft_latents2))
    
    # Combine: low frequencies from original + high frequencies from Gaussian noise
    combined_fft = low_freq_fft + high_freq_fft
    
    # Convert back to spatial domain
    combined_latents = torch.fft.ifft2(torch.fft.ifftshift(combined_fft)).real

    packed_latents = pipe._pack_latents(combined_latents, 1, 16, 128, 128)
    packed_latents = packed_latents.to(torch.bfloat16)

    return packed_latents

def get_latent_statistics(latent):
    print(f"latent mean: {latent.mean()}")
    print(f"latent std: {latent.std()}")

def plot_dict_scatterplot(data_dict, output_path=None, title="Scatterplot", xlabel="X", ylabel="Y", 
                         figsize=(10, 6), color='blue', alpha=0.7, marker='o', markersize=50):
    """
    Create a scatterplot from a dictionary where keys are x-values and values are y-values.
    
    Args:
        data_dict (dict): Dictionary with numeric keys (x-values) and numeric values (y-values)
        output_path (str, optional): Path to save the plot. If None, plot is displayed
        title (str): Title for the plot
        xlabel (str): Label for x-axis
        ylabel (str): Label for y-axis
        figsize (tuple): Figure size (width, height)
        color (str): Color of the scatter points
        alpha (float): Transparency of points (0-1)
        marker (str): Marker style
        markersize (int): Size of markers
    
    Returns:
        fig, ax: matplotlib figure and axis objects
    """
    # Extract x and y values from dictionary
    x_values = list(data_dict.keys())
    y_values = list(data_dict.values())
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatterplot
    scatter = ax.scatter(x_values, y_values, c=color, alpha=alpha, 
                        marker=marker, s=markersize)
    
    # Customize the plot
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add some statistics as text
    if len(x_values) > 0:
        x_range = f"X range: [{min(x_values):.3f}, {max(x_values):.3f}]"
        y_range = f"Y range: [{min(y_values):.3f}, {max(y_values):.3f}]"
        ax.text(0.02, 0.98, f"{x_range}\n{y_range}", 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save or display the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Scatterplot saved to: {output_path}")
    else:
        plt.show()
    
    return fig, ax



def plot_tuple_list_scatterplot(data_list, output_path=None, filename=None,title="Scatterplot", xlabel="X", ylabel="Y", 
                         figsize=(10, 6), color='blue', alpha=0.7, marker='o', markersize=50):
    """
    Create a scatterplot from a dictionary where keys are x-values and values are y-values.
    
    Args:
        data_dict (dict): Dictionary with numeric keys (x-values) and numeric values (y-values)
        output_path (str, optional): Path to save the plot. If None, plot is displayed
        title (str): Title for the plot
        xlabel (str): Label for x-axis
        ylabel (str): Label for y-axis
        figsize (tuple): Figure size (width, height)
        color (str): Color of the scatter points
        alpha (float): Transparency of points (0-1)
        marker (str): Marker style
        markersize (int): Size of markers
    
    Returns:
        fig, ax: matplotlib figure and axis objects
    """
    # Extract x and y values from dictionary
    x_values = [x[0] for x in data_list]
    y_values = [x[1] for x in data_list]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatterplot
    scatter = ax.scatter(x_values, y_values, c=color, alpha=alpha, 
                        marker=marker, s=markersize)
    
    # Customize the plot
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add some statistics as text
    if len(x_values) > 0:
        x_range = f"X range: [{min(x_values):.3f}, {max(x_values):.3f}]"
        y_range = f"Y range: [{min(y_values):.3f}, {max(y_values):.3f}]"
        ax.text(0.02, 0.98, f"{x_range}\n{y_range}", 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save or display the plot

    if output_path and filename:
        output_path = os.path.join(output_path, filename)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_multiple_metrics_scatterplot(metrics_dict, output_path=None, filename=None, title="Multiple Metrics Scatterplot", 
                                     xlabel="Timestep", ylabel="Metric Value", figsize=(12, 8), 
                                     alpha=0.7, marker='o', markersize=50, colors=None):
    """
    Create a scatterplot with multiple metrics from a dictionary of lists of tuples.
    
    Args:
        metrics_dict (dict): Dictionary where keys are metric names and values are lists of tuples (timestep, metric_value)
        output_path (str, optional): Directory path to save the plot. If None, plot is displayed
        filename (str, optional): Filename for the saved plot
        title (str): Title for the plot
        xlabel (str): Label for x-axis
        ylabel (str): Label for y-axis
        figsize (tuple): Figure size (width, height)
        alpha (float): Transparency of points (0-1)
        marker (str): Marker style
        markersize (int): Size of markers
        colors (list, optional): List of colors to use for each metric. If None, uses default color cycle
    
    Returns:
        fig, ax: matplotlib figure and axis objects
    """
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Default colors if not provided
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_dict)))
    
    # Keep track of all x and y values for range calculation
    all_x_values = []
    all_y_values = []
    
    # Plot each metric
    for i, (metric_name, data_list) in enumerate(metrics_dict.items()):
        # Extract x and y values from list of tuples
        x_values = [x[0] for x in data_list]
        y_values = [x[1] for x in data_list]
        
        # Add to overall ranges
        all_x_values.extend(x_values)
        all_y_values.extend(y_values)
        
        # Use color from cycle
        color = colors[i % len(colors)]
        
        # Create scatterplot for this metric
        ax.scatter(x_values, y_values, c=[color], alpha=alpha, 
                  marker=marker, s=markersize, label=metric_name)
    
    # Customize the plot
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add some statistics as text
    if len(all_x_values) > 0:
        x_range = f"X range: [{min(all_x_values):.3f}, {max(all_x_values):.3f}]"
        y_range = f"Y range: [{min(all_y_values):.3f}, {max(all_y_values):.3f}]"
        ax.text(0.02, 0.98, f"{x_range}\n{y_range}", 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save or display the plot
    if output_path and filename:
        full_path = os.path.join(output_path, filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Multiple metrics scatterplot saved to: {full_path}")
    elif output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Multiple metrics scatterplot saved to: {output_path}")
    else:
        plt.show()
    
    return fig, ax

