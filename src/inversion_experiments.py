import os
import re
import time
from dataclasses import dataclass
from glob import iglob
import argparse
import torch
from einops import rearrange
from fire import Fire
from PIL import ExifTags, Image

from flux.sampling import denoise, get_schedule, prepare, unpack, denoise_single_step_to_x0, denoise_starting_particular_step
from flux.util import (configs, embed_watermark, load_ae, load_clip,
                       load_flow_model, load_t5)
from transformers import pipeline
from PIL import Image
import numpy as np
import json
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

@torch.inference_mode()
def latent_to_image(latent, opts, ae):
    batch_latent = unpack(latent.float(), opts.width, opts.height)
    latent = latent.to("cuda")
    for latent in batch_latent:
        latent = latent.unsqueeze(0)
        latent = ae.decode(latent)
        latent = latent.clamp(-1, 1)
        latent = rearrange(latent[0], "c h w -> h w c")
        latent = Image.fromarray((127.5 * (latent + 1.0)).cpu().byte().numpy())
        return latent


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


@torch.inference_mode()
def main(
    args,
    seed: int | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int | None = None,
    loop: bool = False,
    offload: bool = False,
    add_sampling_metadata: bool = True,
    order: int = 2,
):
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

    os.makedirs(output_dir, exist_ok=True)

    nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)

    if name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Got unknown model name: {name}, chose from {available}")

    torch_device = torch.device(device)
    if num_steps is None:
        num_steps = 4 if name == "flux-schnell" else 25

    # init all components
    t5 = load_t5(torch_device, max_length=256 if name == "flux-schnell" else 512)
    clip = load_clip(torch_device)
    model = load_flow_model(name, device="cpu" if offload else torch_device)
    ae = load_ae(name, device="cpu" if offload else torch_device)

    if offload:
        model.cpu()
        torch.cuda.empty_cache()
        ae.encoder.to(torch_device)
    
    init_image = None
    init_image = np.array(Image.open(args.source_img_dir).convert('RGB'))
    init_image_pil = init_image.copy()
    
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

        intermediate_latents = {}
        intermediate_scores = {}
        
        def callback(log):
            t = log["t"]
            intermediate_latents[t] = log["latent"]
            intermediate_scores[t] = log["score"]

        # inversion initial noise

        #! test
        timesteps = timesteps[1:]
        z, info = denoise(model, **inp, timesteps=timesteps, guidance=1, inverse=True, info=info, order=order, callback=callback)
        
        inp_target["img"] = z

        timesteps = get_schedule(opts.num_steps, inp_target["img"].shape[1], shift=(name != "flux-schnell"))

        # denoise initial noise
        x, _ = denoise(model, **inp_target, timesteps=timesteps, guidance=guidance, inverse=False, info=info, order=order)

        # Process denoising
        intermediate_latents_single_step_denoised = {}
        intermediate_latents_denoised = {}

        for i, k in enumerate(sorted(intermediate_latents.keys())):
            if i % 5 == 0 or i == len(intermediate_latents.keys()) - 1 or i >= len(intermediate_latents.keys()) - 5:
                v = intermediate_latents[k]
                v = v.to("cuda")
                inp_target["img"] = v
                img_multi_step, _ = denoise_starting_particular_step(model, **inp_target, timesteps=timesteps, guidance=guidance, target_t=k, inverse=False, info=info, order=order)
                intermediate_latents_denoised[k] = img_multi_step
                img_single_step, _ = denoise_single_step_to_x0(model, **inp_target, guidance=guidance, target_t=k, inverse=False, info=info, order=order)
                intermediate_latents_single_step_denoised[k] = img_single_step

        if offload:
            model.cpu()
            torch.cuda.empty_cache()
            ae.decoder.to(x.device)
        
        psnr_results = {}
        
        for k in intermediate_latents_denoised.keys():
            v = intermediate_latents_denoised[k]
            img = latent_to_image(v, opts, ae)
            img.save(f"{output_dir}/latents_multiple_step_from_{k:02f}.png")
            
            # Compute PSNR between img and init_image
            psnr_value = compute_psnr(img, init_image_pil)
            psnr_results[f"{k:02f}"] = psnr_value
            print(f"PSNR for step {k:02f}: {psnr_value:.2f} dB")

        for k in intermediate_latents_single_step_denoised.keys():
            v = intermediate_latents_single_step_denoised[k]
            img = latent_to_image(v, opts, ae)
            img.save(f"{output_dir}/latents_single_step_from_{k:02f}.png")
        
        # Save PSNR results to JSON file
        psnr_file_path = f"{output_dir}/psnr_results.json"
        with open(psnr_file_path, 'w') as f:
            json.dump(psnr_results, f, indent=2)
        
        print(f"PSNR results saved to: {psnr_file_path}")
        
        # Create and save PSNR scatter plot
        plot_psnr_scatter(psnr_results, output_dir)

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

            fn = output_name.format(idx=idx)
            print(f"Done in {t1 - t0:.1f}s. Saving {fn}")
            # bring into PIL format and save
            x = x.clamp(-1, 1)
            x = embed_watermark(x.float())
            x = rearrange(x[0], "c h w -> h w c")

            z = z.clamp(-1, 1)
            z = rearrange(z[0], "c h w -> h w c")
            z = Image.fromarray((127.5 * (z + 1.0)).cpu().byte().numpy())
            z.save(f"{output_dir}/latents_{idx}.png")

            img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
            nsfw_score = [x["score"] for x in nsfw_classifier(img) if x["label"] == "nsfw"][0]
            
            if nsfw_score < NSFW_THRESHOLD:
                exif_data = Image.Exif()
                exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
                exif_data[ExifTags.Base.Make] = "Black Forest Labs"
                exif_data[ExifTags.Base.Model] = name
                if add_sampling_metadata:
                    exif_data[ExifTags.Base.ImageDescription] = source_prompt
                img.save(fn, exif=exif_data, quality=95, subsampling=0)
                idx += 1
            else:
                print("Your generated image may contain NSFW content.")

            if loop:
                print("-" * 80)
                opts = parse_prompt(opts)
            else:
                opts = None

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
    parser.add_argument('--num_steps', type=int, default=25,
                        help='the number of timesteps for inversion and denoising')
    parser.add_argument('--inject', type=int, default=20,
                        help='the number of timesteps which apply the feature sharing')
    parser.add_argument('--output_dir', default='output', type=str,
                        help='the path of the edited image')
    parser.add_argument('--offload', action='store_true', help='set it to True if the memory of GPU is not enough')
    parser.add_argument('--order', type=int, default=2,
                        help='the order of the diffusion model')

    args = parser.parse_args()

    main(args)
