import torch

model_run_data = torch.load('/home/swhong/workspace/RF-Solver-Edit/FLUX_Image_Edit/src/cache/single_step_inversion/cfe112341087174cfb61cdad0305ed4e_model_run_data.pt', weights_only=False)
single_step_inversion_results = torch.load('/home/swhong/workspace/RF-Solver-Edit/FLUX_Image_Edit/src/cache/single_step_inversion/cfe112341087174cfb61cdad0305ed4e_single_step_results.pt', weights_only=False)


intermediate_latents = model_run_data['intermediate_latents']

initial_noise = model_run_data['initial_noise'].to('cuda')

initial_noise_2 = intermediate_latents[1.0].to('cuda')
initial_estimation = single_step_inversion_results[1.0].to('cuda')

a = torch.norm(initial_estimation - initial_noise, p=2)

print(a)

b = torch.norm(initial_estimation - initial_noise_2, p=2)

print(b)







