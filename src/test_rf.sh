CUDA_VISIBLE_DEVICES=4 python run_rf.py  --source_prompt "A cat holding hello world sign." \
                                        --target_prompt "A cat holding hello world sign." \
                                        --guidance 1 \
                                        --source_img_dir 'examples/source/cat.png' \
                                        --num_steps 30 --offload \
                                        --inject 0 \
                                        --name 'flux-dev'  \
                                        --output_dir 'examples/rf-result/cat'