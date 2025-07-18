import torch
import torch.nn.functional as F
import glob
from torch import Tensor
import math
from einops import rearrange
from src.utils import plot_multiple_metrics_scatterplot, plot_tuple_list_scatterplot

p = "/data/inversion_data/0711_image_data/004027_data.pt"


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


def compute_velocity_curvature(path_to_data):
    data = torch.load(path_to_data, weights_only=False)

    intermediate_latents = data["intermediate_latents"]
    intermediate_scores = data["intermediate_scores"]
    final_latent = data["final_latent"]
    source_latent = data["initial_latent"]
    source_latent = pack(source_latent, 1024, 1024)

    ts = sorted(list(intermediate_latents.keys()))

    from src.utils import get_aligned_generation_intermediate_latents
    aligned_intermediate_latents = get_aligned_generation_intermediate_latents(data)

    # Curvature

    result = {}

    for t in ts:

        # velocity

        velocity = intermediate_scores[t]

        # full_velocity
        full_velocity = (source_latent - final_latent) / (t)

        velocity = velocity.flatten()
        full_velocity = full_velocity.flatten()

        cosine_similarity = F.cosine_similarity(velocity, full_velocity, dim=0)
        mse_diff = torch.mean((velocity - full_velocity) ** 2)

        magnitude_diff = torch.mean((torch.norm(velocity) - torch.norm(full_velocity))**2)

        result[t] = {
            "cosine_similarity": cosine_similarity,
            "mse_diff": mse_diff,
            "magnitude_diff": magnitude_diff,
        }

    return result

def multiprocess_compute_velocity_curvature(path_to_data=None, n_files=100):
    import random
    full_filelists = glob.glob("/data/inversion_data/0712_image_data/*.pt")

    full_filelist = random.sample(full_filelists, n_files)
    
    import multiprocessing as mp
    with mp.Pool(processes=10) as pool:
        results = pool.map(compute_velocity_curvature, full_filelist)
    
    final_dict = {}
    
    # Combine all results into a single dictionary
    for result_dict in results:
        for timestep, metrics in result_dict.items():
            for metric_name, metric_value in metrics.items():
                if metric_name not in final_dict:
                    final_dict[metric_name] = []
                final_dict[metric_name].append((timestep,metric_value.item()))
            
    return final_dict

if __name__ == "__main__":
    results = multiprocess_compute_velocity_curvature(n_files=1000)
    print(f"Processed {len(results)} files")

    plot_tuple_list_scatterplot(results['cosine_similarity'], output_path=".", filename="cosine_similarity.png")
    plot_tuple_list_scatterplot(results['mse_diff'], output_path=".", filename="mse_diff.png")
    plot_tuple_list_scatterplot(results['magnitude_diff'], output_path=".", filename="magnitude_diff.png")
    print("Done")

    # plot_multiple_metrics_scatterplot(results, output_path=".", filename="velocity_curvature_results.png")