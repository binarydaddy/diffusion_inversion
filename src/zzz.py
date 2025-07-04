import torch
from utils import plot_latent_spectrogram, equalize_frequency_magnitudes, normalize_latent, set_frequency_magnitudes_from_gaussian_noise, get_latent_statistics
from inversion import initialize_models, run_model, render_latents_and_compute_psnr, latent_to_image
from dataclasses import dataclass

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

path_to_data = '/home/swhong/workspace/RF-Solver-Edit/FLUX_Image_Edit/src/examples/edit-result/cat/diffusion_data_1.pt'
path_to_image = '/home/swhong/workspace/RF-Solver-Edit/FLUX_Image_Edit/src/examples/source/cat.png'


data = torch.load(path_to_data)

inversion_latents = data['inversion_latents']

ts = sorted(list(inversion_latents.keys()))

last_ts = ts[-1]

original_latent = inversion_latents[last_ts]
# plot_latent_spectrogram(latent, f'cat_ts_{last_ts}_spectrogram.png')

gaussian_noise = torch.randn_like(original_latent)
# plot_latent_spectrogram(gaussian_noise, f'cat_ts_{last_ts}_gaussian_noise_spectrogram.png')

models = initialize_models(name='flux-dev', device='cuda', offload=True)

ae = models['ae']

opts = SamplingOptions(width=1024, height=1024, num_steps=30, guidance=1, seed=None, source_prompt="", target_prompt="")

for freq_ratio in [0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95]:
    latent = set_frequency_magnitudes_from_gaussian_noise(original_latent, frequency_ratio=freq_ratio)

    get_latent_statistics(latent)
    # latent = set_frequency_magnitudes_from_gaussian_noise(latent)
    normalized_latent = normalize_latent(latent)

    rendered_latent = latent_to_image(normalized_latent, opts, ae)
    rendered_latent.save(f'examples/edit-result/cat/latents_mix_freqs_{freq_ratio}.png')

    plot_latent_spectrogram(normalized_latent, f'cat_ts_{last_ts}_normalized_spectrogram.png')

    run_model(path_to_image, latent, models=models, output_filename=f'generated_from_latent_{freq_ratio}.png', device='cuda', name='flux-dev',
               source_prompt='A cat holding hello world sign.', target_prompt='A cat holding hello world sign.', guidance=1, 
               output_dir='examples/edit-result/cat', order=1, num_steps=30, offload=True)