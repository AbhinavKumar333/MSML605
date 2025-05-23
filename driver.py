
import argparse
import torch
from experiments.cpu_batch_sweep import batch_size_sweep
from experiments.optuna_tuning import run_optuna_tuning
from experiments.gpu_batch_sweep import batch_size_sweep_gpu
from experiments.gpu_optuna_tuning import run_gpu_optuna_tuning
from experiments.full_comparisons import run_full_optimization_comparison
from Features.cpu_optimized import train_cpu_model
from Features.gpu_optimized import train_gpu_model
from Utils.caching_results import save_results


RUN_MODULES = {
    "CPU": {"sweep": batch_size_sweep, "tune": run_optuna_tuning, "single": train_cpu_model, "full-compare": run_full_optimization_comparison},
    "GPU": {"sweep": batch_size_sweep_gpu, "tune": run_gpu_optuna_tuning, "single": train_gpu_model, "full-compare": run_full_optimization_comparison}
}


def get_args():
    parser = argparse.ArgumentParser(description="CPU Optimization Benchmark Driver")

    """ Required for every run """
    parser.add_argument('--mode', type=str, choices=['sweep', 'tune', 'single', 'full-compare'], required=True,
                        help='Mode: sweep = batch size sweep, tune = optuna tuning, single = run single config')
    parser.add_argument('--hardware', type=str, choices=['CPU', 'GPU', 'ALL'], required=True,
                        help='Which hardare being used to run')

    """ Only Valid If running in single mode """
    parser.add_argument('--subset', type=bool, default=False, help='Flag to enable reduce dataset size')
    parser.add_argument('--dataset_size', type=int, default=5000, help='Reducing the dataset for quick runs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for single run')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model', type=str, default="simplecnn", 
                        help='Model variant: simplecnn | vgg16 | resnet18 | mobilenetv2')
    return parser.parse_args()


def main():

    args = get_args()

    module = RUN_MODULES[args.hardware.upper()].get(args.mode)
    print("\nRunning {} on {}".format(args.mode, args.hardware))
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))

    if args.mode == 'single':

        result = module(
            subset=args.subset,
            dataset_size=args.dataset_size,
            batch_size=128,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            model_variant=args.model,
            verbose=True
        )

        print("\nSingle Run Result:")
        print(result)
        save_results(args.hardware, result)
    
    elif args.mode == 'full-compare':
        result = module(args.hardware)
        save_results(args.hardware, result)

    else:
        result = module()
        save_results(args.hardware, result)


if __name__ == "__main__":
    main()

# To run the script, use the following command line arguments:

# Full comparison:
# python driver.py --hardware CPU --mode full-compare

# Batch sweep:
# python driver.py --hardware CPU --mode sweep

# Optuna tuning:
# python driver.py --hardare CPU --mode tune

# Single VGG16 training:
# python driver.py --hardware CPU --mode single --model resnet18 --subset True --dataset_size 5000 --batch_size 64 --learning_rate 0.001

# Single ResNet18 training:
# python driver.py --hardware GPU --mode single --model resnet18 --subset True --dataset_size 5000 --batch_size 32 --learning_rate 0.0005

# Docker runs:
# docker run --rm -v /local/data:/data ak395/myapp:cpu-latest