from flux_model_manager import FluxModelManager
import torch
import os
from PIL import Image
import numpy as np
from glob import glob
from utils import compute_psnr, multivariate_gaussian_log_likelihood
# Sanity Check

# Initialize with default settings
def collect_data_for_inversion_timestep_accuracy(prompt="A cat holding a sign that says hello world.", seed=42):

    manager = FluxModelManager(name="flux-dev", device="cuda", offload=False)
    prompt_as_dir = prompt.replace(' ', '_').replace('.', '')
    prompt_as_dir = f"{prompt_as_dir}_{seed}"
    directory = f"output/{prompt_as_dir}"

    if os.path.exists(f"{directory}/generation_final.png") == False:
    
        generation_result = manager.generate(
            prompt=prompt,
            width=1024,
            height=1024,
            num_steps=50,
            guidance=3.5,
            seed=seed,
        )
    
        manager.save_generation_result(
            result=generation_result,
            output_dir=directory,
            filename_prefix="generation"
        )

    # Inversion test

    inversion_timestep_idx = [0, 5, 10, 15, 20, 25, 30]
    for idx in inversion_timestep_idx:
        if os.path.exists(f"{directory}/inverted/inversion_ts_{idx}_data.pt") == False:
            inversion_result = manager.invert_image_starting_from_timestep(
                timestep_idx=idx,
                inversion_data=generation_result,
                prompt=prompt,
            )

            manager.save_inversion_results(
                result=inversion_result,
                output_dir=os.path.join(directory, "inverted"),
                filename_prefix=f"inversion_ts_{idx}"
            )

    # Generation test

    for idx in inversion_timestep_idx:
        inversion_data = torch.load(f"{directory}/inverted/inversion_ts_{idx}_data.pt")
        final_latent = inversion_data["final_latent"]

        if os.path.exists(f"{directory}/generated/generation_ts_{idx}_data.pt") == False:
            generation_result = manager.generate(
                prompt=prompt,
                width=1024,
                height=1024,
                num_steps=50,
                guidance=3.5,
                start_latent=final_latent,
            )

            manager.save_generation_result(
                result=generation_result,
                output_dir=os.path.join(directory, "generated"),
                filename_prefix=f"generation_ts_{idx}"
            )

def check_psnr_for_generation_from_inversion(prompt="A cat holding a sign that says hello world.", seed=42):
    directory = f"output/{prompt.replace(' ', '_').replace('.', '')}_{seed}"

    data_files = glob(os.path.join(directory, "generated", "*.pt"))
    original_data = torch.load(os.path.join(directory, "generation_data.pt"))
    original_img = original_data["final_image"]
    timesteps = original_data["timesteps"]

    psnr_results = {}

    for data_file in data_files:

        timestep_idx = int(data_file.split('/')[-1].split('_')[2])
        timestep = timesteps[timestep_idx]

        data = torch.load(data_file)
        generated_img = data["final_image"]
        psnr = compute_psnr(original_img, generated_img)
        
        psnr_results[timestep] = psnr

    return psnr_results

def check_log_likelihood_for_inversion(prompt="A cat holding a sign that says hello world.", seed=42):
    directory = f"output/{prompt.replace(' ', '_').replace('.', '')}_{seed}"
    original_data = torch.load(os.path.join(directory, "generation_data.pt"))
    original_latent = original_data["final_latent"]
    original_latent = original_latent.reshape(-1)
    timesteps = original_data["timesteps"]

    data_files = glob(os.path.join(directory, "inverted", "*.pt"))

    log_likelihood_results = {}

    log_likelihood_results[0.0] = multivariate_gaussian_log_likelihood(original_latent)

    for data_file in data_files:

        timestep_idx = int(data_file.split('/')[-1].split('_')[2])
        timestep = timesteps[timestep_idx]

        data = torch.load(data_file)
        final_latent = data["final_latent"]
        final_latent = final_latent.reshape(-1)        
        log_likelihood = multivariate_gaussian_log_likelihood(final_latent)

        log_likelihood_results[timestep] = log_likelihood

    return log_likelihood_results


if __name__ == "__main__":
    collect_data_for_inversion_timestep_accuracy()
    ll_results = check_log_likelihood_for_inversion()
    print(ll_results)