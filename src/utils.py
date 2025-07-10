import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math



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

