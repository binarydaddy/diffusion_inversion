import glob
import os
import torch
from utils import plot_tuple_list_scatterplot, plot_multiple_metrics_scatterplot

def main(args):
    validation_result_dir = args.validation_result_dir
    validation_result_files = glob.glob(os.path.join(validation_result_dir, "*.pt"))

    metrics = {}

    for validation_result_file in validation_result_files:        
        sample_name = validation_result_file.split("/")[-1].split("_")[0]

        validation_data = torch.load(validation_result_file, weights_only=False)

        performance_dict = validation_data["performance_dict"]

        for timestep, performance in performance_dict.items():
            for metric_name, metric_value in performance.items():
                if metric_name not in metrics:
                    metrics[metric_name] = []
                metrics[metric_name].append((timestep, metric_value.item()))

    plot_multiple_metrics_scatterplot(metrics, output_path=args.validation_result_dir, \
        filename=f"{metric_name}.png", title=metric_name, xlabel="Timestep", ylabel="Metric Value")

if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("--validation_result_dir", type=str, \
        default="/home/swhong/workspace/diffusion_inversion/src/validation_no_train_result")
    args = parser.parse_args()
    
    main(args)