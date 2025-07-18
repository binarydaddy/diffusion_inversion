from flux_model_manager import FluxModelManager
import torch
import os
from PIL import Image
import numpy as np
from glob import glob
from utils import compute_psnr, multivariate_gaussian_log_likelihood, plot_dict_scatterplot
# Sanity Check

def format_prompt_for_directory(prompt):
    prompt = prompt.replace('\'', '')
    prompt = prompt.replace(' ', '_')
    prompt = prompt.replace('.', '')
    prompt = prompt.replace(',', '')
    prompt = prompt.replace('!', '')
    prompt = prompt.replace('?', '')
    prompt = prompt.replace('"', '')
    prompt = prompt.replace('(', '')
    prompt = prompt.replace(')', '')
    if len(prompt) > 100:
        prompt = prompt[:100]
    return prompt

# Initialize with default settings
def collect_data_for_inversion_timestep_accuracy(prompt="A cat holding a sign that says hello world.", seed=42, device="cuda"):
    """
        This function collects data for the inversion timestep accuracy experiment.
        Step 1: Generate an image from the prompt.
        Step 2: Invert the image starting from a given timestep.
        Step 3: Generate an image from the inverted image starting from the same timestep.
        Then save the data for each step.
    """

    manager = FluxModelManager(name="flux-dev", device=device, offload=False)
    prompt_as_dir = format_prompt_for_directory(prompt)
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

    inversion_timestep_idx = [i for i in range(30) if i % 3 == 0]
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

def check_psnr_for_generation_from_inversion(prompt="A cat holding a sign that says hello world.", seed=42, directory=None):

    if directory is None:
        directory = f"output/{prompt.replace(' ', '_').replace('.', '')}_{seed}"

    data_files = glob(os.path.join(directory, "generated", "*.pt"))
    original_data = torch.load(os.path.join(directory, "generation_data.pt"), weights_only=False)
    original_img = original_data["final_image"]

    timesteps = sorted(original_data["timesteps"])
    psnr_results = {}

    for data_file in data_files:

        timestep_idx = int(data_file.split('/')[-1].split('_')[2])
        timestep = timesteps[timestep_idx]

        data = torch.load(data_file, weights_only=False)
        generated_img = data["final_image"]
        psnr = compute_psnr(original_img, generated_img)
        
        psnr_results[timestep] = psnr

    return psnr_results

def check_log_likelihood_for_inversion(prompt="A cat holding a sign that says hello world.", seed=42, directory=None):

    if directory is None:
        directory = f"output/{prompt.replace(' ', '_').replace('.', '')}_{seed}"    
    
    original_data = torch.load(os.path.join(directory, "generation_data.pt"), weights_only=False)
    original_latent = original_data["final_latent"]
    original_latent = original_latent.reshape(-1)
    timesteps = sorted(original_data["timesteps"])

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

        log_likelihood_results[timestep] = log_likelihood.item()

    return log_likelihood_results

def collect_data_wrapper(caption_path, seed=42, device="cuda"):
    with open(caption_path, "r") as f:
        caption = f.readlines()

    for prompt in caption:
        collect_data_for_inversion_timestep_accuracy(prompt, seed, device)