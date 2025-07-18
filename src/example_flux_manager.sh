python example_flux_manager.py \
  --example inversion_with_timestep_skipping \
  --image_path sample_rome.png \
  --source_prompt "A bustling marketplace in ancient Rome with merchants and citizens." \
  --target_prompt "A bustling marketplace in ancient Rome with merchants and citizens." \
  --output_dir ./output_timestep_skipping/rome \
  --timestep_skipping_lora_path /home/swhong/workspace/diffusion_inversion/trainer/flux_inversion_lora_target_0.3/checkpoint-1000/pytorch_lora_weights.safetensors \
  --timestep_to_skip_to 0.032 \
  --offload