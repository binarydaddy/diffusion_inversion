# CUDA_VISIBLE_DEVICES=0 python inversion.py  --source_prompt "a massive white muscular beast with glowing golden ornaments, large claws and a single sharp horn, adorned with bronze straps and shoulder armor, standing on a rocky pedestal, magical floating horns emitting radiant light beside its body, looking at viewer, 2d game concept art, dynamic lighting, gray background." \
#                                         --target_prompt "a massive white muscular beast with glowing golden ornaments, large claws and a single sharp horn, adorned with bronze straps and shoulder armor, standing on a rocky pedestal, magical floating horns emitting radiant light beside its body, looking at viewer, 2d game concept art, dynamic lighting, gray background." \
#                                         --guidance 1 \
#                                         --source_img_dir 'examples/source/0067_빛의 괴수 피톤.png' \
#                                         --num_steps 30 --offload \
#                                         --inject 0 \
#                                         --name 'flux-dev' \
#                                         --output_dir 'examples/edit-result/0067_빛의 괴수 피톤' \
#                                         --order 1 \
#                                         --run_mode 'single_analysis'

CUDA_VISIBLE_DEVICES=0 python inversion.py  --source_prompt "a female humanoid creature with green hair and flower-like horns, wearing a black and green segmented outfit with organic armor, surrounded by nine large black segmented tails with glowing green blades and claws, standing confidently with one leg raised, looking at viewer, 2d game concept art, dynamic pose, gray background." \
                                        --target_prompt "a female humanoid creature with green hair and flower-like horns, wearing a black and green segmented outfit with organic armor, surrounded by nine large black segmented tails with glowing green blades and claws, standing confidently with one leg raised, looking at viewer, 2d game concept art, dynamic pose, gray background." \
                                        --guidance 1 \
                                        --source_img_dir 'examples/source/0025_아홉 꼬리 블로나.png' \
                                        --num_steps 30 --offload \
                                        --inject 0 \
                                        --name 'flux-dev' \
                                        --output_dir 'examples/edit-result/0025_아홉 꼬리 블로나' \
                                        --order 2 \
                                        --run_mode 'inversion'