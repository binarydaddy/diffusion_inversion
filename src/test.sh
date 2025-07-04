CUDA_VISIBLE_DEVICES=4 python inversion_experiments.py  --source_prompt "A cat holding hello world sign." \
                                        --target_prompt "A cat holding hello world sign." \
                                        --guidance 1 \
                                        --source_img_dir 'examples/source/cat.png' \
                                        --num_steps 30 --offload \
                                        --inject 0 \
                                        --name 'flux-dev' \
                                        --output_dir 'examples/edit-result/cat_exp_t+1' \
                                        --order 1

CUDA_VISIBLE_DEVICES=4 python inversion_experiments.py  --source_prompt "A woman hiking on a trail with mountains in the distance, carrying a backpack and holding a hiking stick." \
                                        --target_prompt "A woman hiking on a trail with mountains in the distance, carrying a backpack and holding a hiking stick." \
                                        --guidance 1 \
                                        --source_img_dir 'examples/source/hiking.jpg' \
                                        --num_steps 30 --offload \
                                        --inject 0 \
                                        --name 'flux-dev' \
                                        --output_dir 'examples/edit-result/hiking_exp_t+1' \
                                        --order 1