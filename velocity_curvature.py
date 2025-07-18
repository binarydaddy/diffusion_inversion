import torch
import torch.nn.functional as F
import glob
from torch import Tensor
import math
from einops import rearrange

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
    final_latent = data["final_latent"]
    source_latent = data["initial_latent"]
    source_latent = pack(source_latent, 1024, 1024)

    ts = sorted(list(intermediate_latents.keys()))

    from src.utils import get_aligned_generation_intermediate_latents
    aligned_intermediate_latents = get_aligned_generation_intermediate_latents(data)

    # Curvature

    result = {}

    for t in ts:

        diff = torch.mean((aligned_intermediate_latents[t] - final_latent) ** 2)

        # velocity
        velocity = (intermediate_latents[t] - final_latent) / (t)

        # full_velocity
        full_velocity = (source_latent - final_latent) / (t)

        velocity = velocity.flatten()
        full_velocity = full_velocity.flatten()

        cosine_similarity = F.cosine_similarity(velocity, full_velocity, dim=0)
        mse_diff = torch.mean((velocity - full_velocity) ** 2)

        result[t] = {
            "cosine_similarity": cosine_similarity,
            "mse_diff": mse_diff,
        }

    return result

def multiprocess_compute_velocity_curvature(path_to_data=None):
    full_filelist = glob.glob("/data/inversion_data/0711_image_data/*.pt")
    
    import multiprocessing as mp
    with mp.Pool(processes=100) as pool:
        results = pool.map(compute_velocity_curvature, full_filelist)
    
    final_dict = {}
    
    # Combine all results into a single dictionary
    for result_dict in results:
        for timestep, metrics in result_dict.items():
            if timestep not in final_dict:
                final_dict[timestep] = {
                    "cosine_similarity": [],
                    "mse_diff": []
                }
            final_dict[timestep]["cosine_similarity"].append(metrics["cosine_similarity"].item())
            final_dict[timestep]["mse_diff"].append(metrics["mse_diff"].item())
    
    return final_dict

if __name__ == "__main__":
    path_to_data = "/data/inversion_data/0711_image_data"
    results = multiprocess_compute_velocity_curvature()
    print(f"Processed {len(results)} files")

    torch.save(results, "velocity_curvature_results.pt")