import torch
import os
import hashlib
import json
from utils import plot_latent_spectrogram, equalize_frequency_magnitudes, normalize_latent, set_frequency_magnitudes_from_gaussian_noise, get_latent_statistics
from inversion import initialize_models, run_model, render_latents_and_compute_psnr_and_compute_likelihood, latent_to_image, plot_l2_scatter
from dataclasses import dataclass
from flux.sampling import invert_single_step
import random

@dataclass
class SamplingOptions:
    source_prompt: str
    target_prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None

class SingleStepInversionAnalyzer:
    def __init__(self, cache_dir="cache", base_output_dir="examples/edit-result"):
        self.cache_dir = cache_dir
        self.base_output_dir = base_output_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def _generate_cache_key(self, seed, source_prompt, target_prompt, guidance, num_steps, order, path_to_image):
        """Generate a unique cache key based on parameters."""
        params = {
            'seed': seed,
            'source_prompt': source_prompt,
            'target_prompt': target_prompt,
            'guidance': guidance,
            'num_steps': num_steps,
            'order': order,
            'path_to_image': path_to_image
        }
        # Create hash from parameters
        params_str = json.dumps(params, sort_keys=True)
        cache_key = hashlib.md5(params_str.encode()).hexdigest()
        return cache_key
    
    def _get_cache_paths(self, cache_key):
        """Get cache file paths for a given cache key."""
        return {
            'model_run_data': os.path.join(self.cache_dir, f"{cache_key}_model_run_data.pt"),
            'single_step_results': os.path.join(self.cache_dir, f"{cache_key}_single_step_results.pt"),
            'metadata': os.path.join(self.cache_dir, f"{cache_key}_metadata.json")
        }
    
    def _save_to_cache(self, cache_key, model_run_data, single_step_results, metadata):
        """Save results to cache."""
        cache_paths = self._get_cache_paths(cache_key)
        
        # Save model run data
        torch.save(model_run_data, cache_paths['model_run_data'])
        
        # Save single step results
        torch.save(single_step_results, cache_paths['single_step_results'])
        
        # Save metadata
        with open(cache_paths['metadata'], 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Results cached with key: {cache_key}")
    
    def _load_from_cache(self, cache_key):
        """Load results from cache if available."""
        cache_paths = self._get_cache_paths(cache_key)
        
        # Check if all cache files exist
        if all(os.path.exists(path) for path in cache_paths.values()):
            print(f"Loading cached results for key: {cache_key}")
            
            model_run_data = torch.load(cache_paths['model_run_data'], weights_only=False)
            single_step_results = torch.load(cache_paths['single_step_results'], weights_only=False)
            
            with open(cache_paths['metadata'], 'r') as f:
                metadata = json.load(f)
            
            return model_run_data, single_step_results, metadata
        
        return None, None, None
    
    def run_model_with_caching(self, path_to_image, gaussian_noise, models, seed, source_prompt, target_prompt, 
                              guidance, num_steps, order, output_dir_suffix=""):
        """Run model with caching support."""
        
        # Generate cache key
        cache_key = self._generate_cache_key(
            seed=seed,
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            guidance=guidance,
            num_steps=num_steps,
            order=order,
            path_to_image=path_to_image
        )
        
        # Try to load from cache
        model_run_data, single_step_results, metadata = self._load_from_cache(cache_key)
        
        if model_run_data is not None:
            print("✅ Using cached results")
            return model_run_data, single_step_results, metadata
        
        print("🔄 Computing new results...")
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Create output directory
        output_dir = f'{self.base_output_dir}/cat_single_step_inversion_error_test{output_dir_suffix}'
        
        # Run the model
        model_run_data = run_model(
            path_to_image, gaussian_noise, models=models, 
            output_filename=f'generated_from_latent.png', 
            device='cuda', name='flux-dev',
            source_prompt=source_prompt, 
            target_prompt=target_prompt, 
            guidance=guidance, 
            output_dir=output_dir, 
            order=order, 
            num_steps=num_steps, 
            offload=False,
        )

        # Perform single step inversion analysis
        single_step_results = self._perform_single_step_inversion(model_run_data, models, gaussian_noise)
        
        model_run_data['initial_noise'] = gaussian_noise.cpu()

        # Create metadata
        metadata = {
            'cache_key': cache_key,
            'seed': seed,
            'source_prompt': source_prompt,
            'target_prompt': target_prompt,
            'guidance': guidance,
            'num_steps': num_steps,
            'order': order,
            'path_to_image': path_to_image,
            'output_dir': output_dir,
            'timestamp': torch.tensor(0).item(),  # Simple timestamp
        }
        
        # Save to cache
        self._save_to_cache(cache_key, model_run_data, single_step_results, metadata)
        
        return model_run_data, single_step_results, metadata
    
    def _perform_single_step_inversion(self, model_run_data, models, gaussian_noise):
        """Perform single step inversion analysis."""
        intermediate_latents = model_run_data['intermediate_latents']
        timesteps = model_run_data['timesteps']
        
        # Get the input parameters from model_run_data
        inp = model_run_data['inputs']
        
        info = {
            'feature_path': '',
            'feature': {},
            'inject_step': 0
        }

        single_step_inversion_results = {}

        latents = {}
        scores = {}
        
        def callback(log):
            latents[log['t']] = log['latent'].cpu()
            scores[log['t']] = log['score'].cpu()

        for i, t in enumerate(timesteps):
            if t == 0.0:
                continue
            current_latents = intermediate_latents[t]
            inverted_latent, _ = invert_single_step(
                model=models['model'],
                img=current_latents.to('cuda'),
                img_ids=inp['img_ids'].to('cuda'),        # Use from inp dictionary
                txt=inp['txt'].to('cuda'),                # Use from inp dictionary
                txt_ids=inp['txt_ids'].to('cuda'),        # Use from inp dictionary
                vec=inp['vec'].to('cuda'),                # Use from inp dictionary
                target_t=t,
                timesteps=timesteps,
                inverse=True,
                info=info,
                guidance=1,
                order=1,
                callback=callback
            )
            
        single_step_inversion_results['intermediate_latents'] = latents
        single_step_inversion_results['intermediate_scores'] = scores

        return single_step_inversion_results
    
    def analyze_score_differences(self, model_run_scores, single_step_scores, output_dir):
        """Analyze score differences and create plots."""
        
        import torch.nn.functional as F
        import matplotlib.pyplot as plt
        
        cosine_sim_dict = {}
        
        # Find common timesteps between both score dictionaries
        common_timesteps = set(model_run_scores.keys()) & set(single_step_scores.keys())
        common_timesteps = sorted(list(common_timesteps), reverse=True)
        
        print(f"Computing cosine similarity for {len(common_timesteps)} common timesteps...")
        
        for t in common_timesteps:
            model_score = model_run_scores[t]
            single_step_score = single_step_scores[t]
            
            # Ensure both scores are on the same device
            device = model_score.device if model_score.device != torch.device('cpu') else single_step_score.device
            model_score = model_score.to(device)
            single_step_score = single_step_score.to(device)
            
            # Flatten the tensors for cosine similarity computation
            model_score_flat = model_score.flatten()
            single_step_score_flat = single_step_score.flatten()
            
            # Compute cosine similarity
            cosine_sim = F.cosine_similarity(
                model_score_flat.unsqueeze(0), 
                single_step_score_flat.unsqueeze(0), 
                dim=1
            ).item()
            
            cosine_sim_dict[t] = cosine_sim
            print(f"Cosine similarity at t={t}: {cosine_sim:.6f}")
        
        # Create cosine similarity plot
        self._plot_cosine_similarity_scatter(cosine_sim_dict, output_dir)
        
        return cosine_sim_dict
    
    def _plot_cosine_similarity_scatter(self, cosine_sim_dict, output_dir, plotname="cosine_similarity_scatter_plot"):
        """Create a scatter plot of cosine similarity values."""
        import matplotlib.pyplot as plt
        
        # Prepare data for plotting
        timesteps = list(cosine_sim_dict.keys())
        cosine_sims = list(cosine_sim_dict.values())
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.scatter(timesteps, cosine_sims, alpha=0.7, s=50)
        plt.xlabel('Timestep')
        plt.ylabel('Cosine Similarity')
        plt.title('Cosine Similarity between Model Run Scores and Single Step Scores')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, max(timesteps) * 1.1)
        plt.ylim(min(cosine_sims) * 0.9, max(cosine_sims) * 1.1)
        
        # Add horizontal line at cosine similarity = 1 (perfect similarity)
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect Similarity')
        plt.legend()
        
        # Save the plot
        plot_path = os.path.join(output_dir, f'{plotname}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Cosine similarity plot saved to: {plot_path}")
        
        # Print summary statistics
        print(f"\n📊 Cosine Similarity Summary:")
        print(f"   Mean: {sum(cosine_sims) / len(cosine_sims):.6f}")
        print(f"   Min:  {min(cosine_sims):.6f}")
        print(f"   Max:  {max(cosine_sims):.6f}")
        print(f"   Std:  {(sum([(x - sum(cosine_sims) / len(cosine_sims))**2 for x in cosine_sims]) / len(cosine_sims))**0.5:.6f}")
    
    def analyze_l2_differences(self, model_run_data, single_step_results, output_dir, device='cpu'):
        """Analyze L2 differences and create plots."""
        
        gt_latents = model_run_data['gt_latents']
    
        diff_dict = {}
        directional_diff_dict = {}

        single_step_latents = single_step_results['intermediate_latents']

        timesteps = sorted(list(single_step_latents.keys()), reverse=True)

        for i, k in enumerate(timesteps):
            if i == len(timesteps) - 1:
                next_ts = 0.0
            else:
                next_ts = timesteps[i+1]
            dt = k - next_ts
            l2_diff = torch.norm(single_step_latents[k].to(device) - gt_latents[k].to(device), p=2)
            diff_dict[k] = l2_diff.item()  # Convert to Python float for JSON serialization
            directional_diff_dict[k] = l2_diff.item() / dt
            print(f"L2 diff at {k}: {l2_diff}")
        
        # Create L2 scatter plot
        plot_l2_scatter(diff_dict, output_dir)
        plot_l2_scatter(directional_diff_dict, output_dir, plotname="directional_l2_scatter_plot")
        
        return diff_dict
    
    @torch.inference_mode()
    def render_single_step_results(self, single_step_results, models, metadata, output_dir, child_dir_name="single_step_renderings"):
        """
        Render images from single step inversion results using multi-step denoising.
        
        Args:
            single_step_results (dict): Dictionary with timestep keys and latent values
            models (dict): Dictionary containing all models
            metadata (dict): Metadata containing sampling options
            output_dir (str): Directory to save rendered images
        """
        from flux.sampling import denoise_starting_particular_step, get_schedule
        from flux.sampling import prepare
        from inversion import encode
        import numpy as np
        from PIL import Image
        
        print("🎨 Rendering single step inversion results...")
        
        # Create rendering output directory
        render_dir = os.path.join(output_dir, child_dir_name)
        os.makedirs(render_dir, exist_ok=True)
        
        # Reconstruct sampling options from metadata
        opts = SamplingOptions(
            source_prompt=metadata['source_prompt'],
            target_prompt=metadata['target_prompt'],
            width=1024,  # Default width
            height=1024,  # Default height
            num_steps=metadata['num_steps'],
            guidance=metadata['guidance'],
            seed=metadata['seed']
        )
        
        # Get models
        model = models['model']
        ae = models['ae']
        t5 = models['t5']
        clip = models['clip']
        torch_device = models['torch_device']
        
        # Load and encode the original image for text preparation
        init_image = np.array(Image.open(metadata['path_to_image']).convert('RGB'))
        shape = init_image.shape
        new_h = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
        new_w = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16
        init_image = init_image[:new_h, :new_w, :]
        init_image_encoded = encode(init_image, torch_device, ae)
        
        # Prepare text embeddings using target prompt
        inp = prepare(t5, clip, init_image_encoded, prompt=opts.target_prompt)
        
        # Get timestep schedule
        timesteps = get_schedule(opts.num_steps, list(single_step_results.values())[0].shape[1], shift=True)
        
        # Setup info dictionary
        info = {
            'feature_path': 'feature',
            'feature': {},
            'inject_step': 0
        }
        
        print(f"📊 Rendering {len(single_step_results)} latents...")
        
        # Process each latent in single_step_results
        for target_t, starting_latent in single_step_results.items():
            print(f"🔄 Rendering from timestep {target_t}")
            
            # Prepare input with the starting latent
            inp_with_latent = inp.copy()
            inp_with_latent["img"] = starting_latent.to(torch_device)
            
            # Perform multi-step denoising starting from target_t
            try:
                final_latent, _ = denoise_starting_particular_step(
                    model, 
                    **inp_with_latent, 
                    timesteps=timesteps, 
                    target_t=target_t, 
                    guidance=opts.guidance, 
                    inverse=False, 
                    info=info, 
                    order=metadata['order']
                )
                
                # Decode to image
                from flux.sampling import unpack
                from einops import rearrange
                
                batch_x = unpack(final_latent.float(), opts.width, opts.height)
                
                with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                    x = ae.decode(batch_x)
                
                # Convert to PIL image
                x = x.clamp(-1, 1)
                x = rearrange(x[0], "c h w -> h w c")
                img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
                
                # Save the rendered image
                output_path = f"{render_dir}/single_step_t_{target_t:.3f}.png"
                img.save(output_path)
                print(f"✅ Saved: {output_path}")
                
            except Exception as e:
                print(f"❌ Error rendering timestep {target_t}: {e}")
        
        print(f"🎉 Rendering complete! Images saved to: {render_dir}")
        return render_dir

def main():
    """Main function to run the single step inversion analysis."""
    
    # Configuration
    path_to_data = '/home/swhong/workspace/RF-Solver-Edit/FLUX_Image_Edit/src/examples/edit-result/cat/diffusion_data_1.pt'
    path_to_image = '/home/swhong/workspace/RF-Solver-Edit/FLUX_Image_Edit/src/examples/source/cat.png'
    
    # Parameters
    seed = 2
    source_prompt = 'A cat holding hello world sign.'
    target_prompt = 'A cat holding hello world sign.'
    guidance = 1
    num_steps = 30
    order = 1
    
    # Load original data
    data = torch.load(path_to_data)
    inversion_latents = data['inversion_latents']
    ts = sorted(list(inversion_latents.keys()))
    last_ts = ts[-1]
    original_latent = inversion_latents[last_ts]
    
    # Generate gaussian noise with seed
    torch.manual_seed(seed)
    gaussian_noise = torch.randn_like(original_latent)
    
    # Initialize models
    models = initialize_models(name='flux-dev', device='cuda', offload=False)
    
    # Initialize analyzer with caching
    analyzer = SingleStepInversionAnalyzer(
        cache_dir="cache/single_step_inversion",
        base_output_dir="examples/edit-result"
    )
    
    # Run analysis with caching
    model_run_data, single_step_results, metadata = analyzer.run_model_with_caching(
        path_to_image=path_to_image,
        gaussian_noise=gaussian_noise,
        models=models,
        seed=seed,
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        guidance=guidance,
        num_steps=num_steps,
        order=order,
        output_dir_suffix="_cached"
    )
    
    model_run_data['gt_latents'] = realign_model_run_data(model_run_data)
 
    # Analyze L2 differences
    output_dir = metadata.get('output_dir', 'examples/edit-result/cat_single_step_inversion_error_test_cached')
    diff_dict = analyzer.analyze_l2_differences(model_run_data, single_step_results, output_dir)

    score_diff_dict = analyzer.analyze_score_differences(single_step_results['intermediate_scores'], model_run_data['intermediate_scores'], output_dir)
    
    print(f"\n✅ L2 Analysis complete!")
    print(f"📊 L2 differences computed for {len(diff_dict)} timesteps")
    print(f"📊 Score differences computed for {len(score_diff_dict)} timesteps")
    print(f"📁 Results saved to: {output_dir}")
    
    # Render single step inversion results
    print(f"\n🎨 Starting rendering of inversion single step results...")
    render_dir = analyzer.render_single_step_results(single_step_results['intermediate_latents'], models, metadata, output_dir)

    print(f"\n🎨 Starting rendering of forward latent results...")
    forward_dir = analyzer.render_single_step_results(model_run_data['gt_latents'], models, metadata, output_dir, child_dir_name="forward_latent_renderings")

    
    print(f"\n🎉 Complete analysis finished!")
    print(f"📊 L2 differences: {len(diff_dict)} timesteps")
    print(f"🖼️  Rendered images: {len(single_step_results)} images")
    print(f"📁 L2 results: {output_dir}")
    print(f"📁 Rendered images: {render_dir}")
    print(f"📁 Forward Rendered images: {forward_dir}")

def realign_model_run_data(model_run_data):
    
    # These are latents obtained from diffusion denoising process.
    intermediate_latents_timesteps = sorted(list(model_run_data['intermediate_latents'].keys()), reverse=True)
    intermediate_latents = model_run_data['intermediate_latents']

    gt_latents = {}
    for i, k in enumerate(intermediate_latents_timesteps):
        
        if i == len(intermediate_latents_timesteps) - 1:
            continue
        
        v = intermediate_latents[k]
        next_ts = intermediate_latents_timesteps[i+1]
        gt_latents[next_ts] = v

    gt_latents[1.0] = model_run_data['initial_noise']

    return gt_latents


if __name__ == "__main__":
    main()
