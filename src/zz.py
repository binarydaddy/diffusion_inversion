from flux_model_manager import FluxModelManager

# Initialize with default settings
manager = FluxModelManager(name="flux-dev", device="cuda", offload=False)

result = manager.generate(
    prompt="a beautiful sunset over mountains",
    width=1024,
    height=1024,
    num_steps=50,
    guidance=3.5,
    seed=42,
    save_intermediates=True
)

# Save the results
manager.save_generation_result(result, "output", "sunset_generation")
