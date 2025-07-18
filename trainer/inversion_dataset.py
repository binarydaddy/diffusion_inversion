from glob import glob
import json
import torch
from torch.utils.data import Dataset

class InversionDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        data_root="/data/inversion_data/0711_image_data",
        image_size=1024,
        target_timestep=0.3,
        timestep_infer_idx=0,
    ):

        self.data_files = self.parse_data_root(data_root)
        self._length = len(self.data_files)
        self._data_root = data_root
        self.target_timestep = target_timestep
        self.image_size = image_size

        # Change timestep_infer_idx to 0, if we want to start inversion from 0.0.
        # Current implementation is x_(t+1) = x_t + f(x_t, t+1)

        self.timestep_infer_idx = timestep_infer_idx

    def parse_data_root(self, data_root):
        all_data_names = []
        for file in sorted(glob(f"{data_root}/*.pt")):
            all_data_names.append(file.split("/")[-1].split(".")[0].split("_")[0])

        return all_data_names

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        
        current_data_name = self.data_files[index]
        data_path_prefix = f"{self._data_root}"
        
        # prompt
        with open(f"{data_path_prefix}/{current_data_name}_metadata.json", "r") as f:
            metadata = json.load(f)
        
        prompt = metadata["prompt"]
        
        # image
        intermediate_data = torch.load(f"{data_path_prefix}/{current_data_name}_data.pt", weights_only=False)

        source_latents = intermediate_data["final_latent"]
        # target

        intermediate_latents = intermediate_data["intermediate_latents"]
        
        timestep_list = sorted(list(intermediate_latents.keys()))
        timestep_to_infer = timestep_list[self.timestep_infer_idx]

        for i, timestep in enumerate(timestep_list):
            if timestep >= self.target_timestep:
                break

        target_latents = intermediate_latents[timestep]
        
        if i > 0:
            timestep_denominator = timestep_list[i-1]
        else:
            timestep_denominator = timestep

        velocity = (target_latents - source_latents) / timestep_denominator

        # print(f"-----INFO-----")
        # print(f"timestep_to_infer = {timestep_to_infer}")
        # print(f"timesteps = {timestep_list}")
        # print(f"start_timestep = {timestep}")
        # exit()

        return {
            "prompt": prompt,
            "source_latents": source_latents.squeeze(0),
            "target_latents": target_latents.squeeze(0),
            "velocity": velocity.squeeze(0),
            "timestep_to_infer": timestep_to_infer,
        }

    def compute_velocity(self, source_latents, target_latents, timestep_denominator):
        return (target_latents - source_latents) / timestep_denominator



class RandomTimestepAccessInversionDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        data_root="/data/inversion_data/0711_image_data",
        image_size=1024,
        target_timestep=0.3,
        timestep_infer_idx=0,
    ):

        self.data_files = self.parse_data_root(data_root)
        self._length = len(self.data_files)
        self._data_root = data_root
        self.target_timestep = target_timestep
        self.image_size = image_size

        # Change timestep_infer_idx to 0, if we want to start inversion from 0.0.
        # Current implementation is x_(t+1) = x_t + f(x_t, t+1)

        self.timestep_infer_idx = timestep_infer_idx

    def parse_data_root(self, data_root):
        all_data_names = []
        for file in sorted(glob(f"{data_root}/*.pt")):
            all_data_names.append(file.split("/")[-1].split(".")[0].split("_")[0])

        return all_data_names

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        
        current_data_name = self.data_files[index]
        data_path_prefix = f"{self._data_root}"
        
        # prompt
        with open(f"{data_path_prefix}/{current_data_name}_metadata.json", "r") as f:
            metadata = json.load(f)
        
        prompt = metadata["prompt"]
        
        # image
        intermediate_data = torch.load(f"{data_path_prefix}/{current_data_name}_data.pt", weights_only=False)

        source_latents = intermediate_data["final_latent"]
        # target

        intermediate_latents = intermediate_data["intermediate_latents"]

        return {
            "prompt": prompt,
            "source_latents": source_latents.squeeze(0),
            "intermediate_latents": intermediate_latents,
        }

    def collate_fn(self, examples):
        prompts = [example["prompt"] for example in examples]
        source_latents = [example["source_latents"] for example in examples]
        intermediate_latents_list = [example["intermediate_latents"] for example in examples]

        batched_intermediate_latents = {}
        for timestep, intermediate_latent in intermediate_latents_list:
            if timestep not in batched_intermediate_latents:
                batched_intermediate_latents[timestep] = []
            batched_intermediate_latents[timestep].append(intermediate_latent)

        for timestep, intermediate_latent in batched_intermediate_latents.items():
            batched_intermediate_latents[timestep] = torch.stack(intermediate_latent)
            batched_intermediate_latents[timestep] = batched_intermediate_latents[timestep].to(memory_format=torch.contiguous_format).float()

        return {
            "prompts": prompts,
            "source_latents": source_latents,
            "intermediate_latents": batched_intermediate_latents,
        }