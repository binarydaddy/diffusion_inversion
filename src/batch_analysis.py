from research_experiments import check_psnr_for_generation_from_inversion, check_log_likelihood_for_inversion
from utils import plot_tuple_list_scatterplot, plot_dict_scatterplot
import glob
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def make_psnr_log_likelihood_scatterplots(output_dir="output"):

    # PSNR, Log Likelihood ScatterPlots
    full_psnr_results = []
    full_log_likelihood_results = []

    psnr_results = check_psnr_for_generation_from_inversion(directory=output_dir)
    log_likelihood_results = check_log_likelihood_for_inversion(directory=output_dir)

    for k, v in psnr_results.items():
        full_psnr_results.append((k, v))

    for k, v in log_likelihood_results.items():
        full_log_likelihood_results.append((k, v))

    plot_tuple_list_scatterplot(full_psnr_results, output_dir, "psnr_scatterplot.png", "PSNR", ylabel="PSNR")
    plot_tuple_list_scatterplot(full_log_likelihood_results, output_dir, "log_likelihood_scatterplot.png", "Log Likelihood", ylabel="Log Likelihood")

def make_image_grid(output_dir="output"):

    original_generated_image_path = os.path.join(output_dir, "generation_final.png")

    p = os.path.join(output_dir, "generated", "*.png")

    all_files = glob.glob(p)
    idxs = [int(file.split('/')[-1].split('.')[0].split('_')[-2]) for file in all_files]

    files_with_idxs = list(zip(all_files, idxs))
    files_with_idxs.sort(key=lambda x: x[1])

    # Load the original generated image
    if not os.path.exists(original_generated_image_path):
        print(f"Warning: Original generated image not found at {original_generated_image_path}")
        return
    
    original_img = Image.open(original_generated_image_path)
    
    # Load all timestep images
    timestep_images = []
    timestep_labels = []
    
    for sample, timestep_idx in files_with_idxs:
        if os.path.exists(sample):
            img = Image.open(sample)
            timestep_images.append(img)
            timestep_labels.append(f"t={timestep_idx}")
    
    if not timestep_images:
        print("No timestep images found")
        return
    
    # Create the plot
    n_timesteps = len(timestep_images)
    fig, axes = plt.subplots(2, max(1, n_timesteps), figsize=(4*max(1, n_timesteps), 8))
    
    # Handle case where there's only one column
    if n_timesteps == 1:
        axes = axes.reshape(2, 1)
    
    # Plot original image in the first row (centered if there are multiple columns)
    if n_timesteps > 1:
        # Center the original image across multiple columns
        for i in range(n_timesteps):
            axes[0, i].axis('off')
        
        # Use the middle subplot for the original image
        middle_idx = n_timesteps // 2
        axes[0, middle_idx].imshow(original_img)
        axes[0, middle_idx].set_title("Original Generated Image", fontsize=12, fontweight='bold')
        axes[0, middle_idx].axis('off')
    else:
        # Single column case
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title("Original Generated Image", fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
    
    # Plot timestep images in the second row
    for i, (img, label) in enumerate(zip(timestep_images, timestep_labels)):
        axes[1, i].imshow(img)
        axes[1, i].set_title(label, fontsize=10)
        axes[1, i].axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, "inversion_timestep_geneneration_plot.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    
    plt.close()

def main(output_path):
    make_image_grid(output_path)
    make_psnr_log_likelihood_scatterplots(output_path)

if __name__ == "__main__":
    main("/home/swhong/workspace/diffusion_inversion/src/output/Tribal_mask_with_intricate_carved_patterns_and_bright_colors\n_42")