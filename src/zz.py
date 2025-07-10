from flux_model_manager import FluxModelManager

# Initialize with default settings
manager = FluxModelManager(name="flux-dev", device="cuda", offload=False)

# result = manager.generate(
#     prompt="a beautiful sunset over mountains",
#     width=1024,
#     height=1024,
#     num_steps=50,
#     guidance=3.5,
#     seed=42,
#     save_intermediates=True
# )

# # Save the results
# manager.save_generation_result(result, "output", "sunset_generation")

inversion_result = manager.invert_image(
    image_path="output/sunset_generation_final.png",
    source_prompt="a beautiful sunset over mountains.",
    num_steps=50,
    guidance=3.5,
    order=2,
    save_intermediates=True,
)

manager.save_inversion_results(inversion_result, "output", "sunset_inversion")

gen_result = manager.generate(
    prompt="a beautiful sunset over mountains.",
    starting_latent=inversion_result.final_latent,
    width=1024,
    height=1024,
    num_steps=50,
    guidance=3.5,
    order=2,
)

manager.save_generation_result(gen_result, "output", "sunset_generation_from_inversion")