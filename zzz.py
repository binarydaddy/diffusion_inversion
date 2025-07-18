from diffusers import FluxTransformer2DModel

p = "black-forest-labs/FLUX.1-dev"
a = FluxTransformer2DModel.from_pretrained(p, subfolder="transformer")

with open('flux_params.txt', 'w') as f:
    for name, param in a.named_parameters():
        f.write(f"{name}:{param.shape}\n")
