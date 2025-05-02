
import argparse
from experiments.cpu_batch_sweep import batch_size_sweep
from experiments.optuna_tuning import run_optuna_tuning
from experiments.gpu_batch_sweep import batch_size_sweep_gpu
from experiments.gpu_optuna_tuning import run_gpu_optuna_tuning
from Features.cpu_optimized import train_cpu_model
from Features.gpu_optimized import train_gpu_model
from Utils.caching_results import save_results


RUN_MODULES = {
    "CPU": {"sweep": batch_size_sweep, "tune": run_optuna_tuning, "single": train_cpu_model},
    "GPU": {"sweep": batch_size_sweep_gpu, "tune": run_gpu_optuna_tuning, "single": train_gpu_model},
}


def get_args():
    parser = argparse.ArgumentParser(description="CPU Optimization Benchmark Driver")

    """ Required for every run """
    parser.add_argument('--mode', type=str, choices=['sweep', 'tune', 'single'], required=True,
                        help='Mode: sweep = batch size sweep, tune = optuna tuning, single = run single config')
    parser.add_argument('--hardware', type=str, choices=['CPU', 'GPU'], required=True,
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
    
    else:
        
        result = module(subset=args.subset, dataset_size=args.dataset_size, epochs=args.epochs, learning_rate=args.learning_rate)
        save_results(args.hardware, result)


if __name__ == "__main__":
    main()


# Batch sweep:
# python driver.py --hardware CPU --mode sweep

# Optuna tuning:
# python driver.py --hardare CPU --mode tune

# Single VGG16 training:
# python driver.py --hardware CPU --mode single --model vgg16 --subset True --dataset_size 5000 --batch_size 64 --learning_rate 0.001

# Single ResNet18 training:
# python driver.py --hardware GPU --mode single --model resnet18 --subset True --dataset_size 5000 --batch_size 32 --learning_rate 0.0005