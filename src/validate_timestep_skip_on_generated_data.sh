python validate_timestep_skip_on_generated_data.py \
--timestep_skipping_lora_path /home/swhong/workspace/diffusion_inversion/trainer/flux_inversion_lora_target_0.3_512/checkpoint-3000/pytorch_lora_weights.safetensors \
--timestep_to_skip_to 0.31 \
--generation_data_dir /home/swhong/workspace/diffusion_inversion/src/validation_no_train \
--validated_result_dir /home/swhong/workspace/diffusion_inversion/src/validation_no_train_result \
--device cuda \
--offload True