from flux.rf_inversion import RFInversionFluxPipeline
from PIL import Image
import os
import torch

def rf_inversion(args):

    os.makedirs(args.output_dir, exist_ok=True)

    pipe = RFInversionFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()

    image = Image.open(args.source_img_dir)        
    height, width = image.size

    inverted_latents, image_latents, latent_image_ids, latent_image = pipe.invert(
        image=image,
        source_prompt=args.source_prompt,
        source_guidance_scale=args.guidance,
        num_inversion_steps=args.num_steps,
        strength=1.0,
        gamma=0.0,
        height=height,
        width=width,
    )

    # inverted_latents = normalize_latents(inverted_latents)
    latent_image[0].save(f"{args.output_dir}/image_latents.png")

    result = pipe(
        prompt=args.target_prompt,
        inverted_latents=inverted_latents,
        image_latents=image_latents,
        latent_image_ids=latent_image_ids,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance,
        eta=0.0,
    ).images[0]

    result.save(f"{args.output_dir}/image_gs_{args.guidance}.png")

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--source_img_dir", type=str, default="examples/source/cat.png")
    parser.add_argument("--source_prompt", type=str, default="A cat holding hello world sign.")
    parser.add_argument("--target_prompt", type=str, default="A cat holding hello world sign.")
    parser.add_argument("--guidance", type=float, default=0)
    parser.add_argument("--num_steps", type=int, default=30)
    parser.add_argument("--offload", action='store_true')
    parser.add_argument("--inject", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="examples/edit-result/cat")
    parser.add_argument("--name", type=str, default="flux-dev")
    args = parser.parse_args()
    rf_inversion(args)