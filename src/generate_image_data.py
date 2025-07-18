from flux_model_manager import FluxModelManager
import os

def generate_image_data(prompt_path, output_dir, seed=None, device="cuda", chunk_idx=0, order=2):
    """
    Generate image data from prompts in a file.
    """
    
    manager = FluxModelManager(name="flux-dev", device=device, offload=False)
    
    with open(prompt_path, "r") as f:
        prompts = f.readlines()
    
    start_idx = chunk_idx * 5000

    for i, prompt in enumerate(prompts):

        current_idx = start_idx + i
        if os.path.exists(f"{output_dir}/{current_idx:06d}"):
            continue

        prompt = prompt.strip()
        
        generation_result = manager.generate(
            prompt=prompt,
            seed=seed,
            width=1024,
            height=1024,
            num_steps=50,
            guidance=3.5,
            order=order,
        )
        
        manager.save_generation_result(
            result=generation_result,
            output_dir=output_dir,
            filename_prefix=f"{current_idx:06d}",
        )

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--order", type=int, default=2)
    args = parser.parse_args()
    
    generate_image_data(args.prompt_path, args.output_dir, args.seed, args.device, args.chunk_idx, args.order)