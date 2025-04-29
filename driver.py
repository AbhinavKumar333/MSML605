import argparse
from experiments.cpu_batch_sweep import batch_size_sweep
from experiments.optuna_tuning import run_optuna_tuning

def main():
    parser = argparse.ArgumentParser(description="CPU Optimization Benchmark Driver")
    parser.add_argument('--mode', type=str, choices=['sweep', 'tune', 'single'], required=True,
                        help='Mode: sweep = batch size + model variant sweep, tune = optuna hyperparameter search, single = run single config')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for single run')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model', type=str, default="vgg16", help='Model variant: vgg16 | resnet18 | mobilenetv2')

    args = parser.parse_args()

    if args.mode == 'sweep':
        batch_size_sweep()

    elif args.mode == 'tune':
        run_optuna_tuning()

    elif args.mode == 'single':
        from features.cpu_optimized import train_cpu_model
        result = train_cpu_model(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            model_variant=args.model,
            verbose=True,
            quantize=False
        )
        print("\nSingle Run Result:")
        print(result)

if __name__ == "__main__":
    main()


# Batch sweep:
# python driver.py --mode sweep

# Optuna tuning:
# python driver.py --mode tune

# Single VGG16 training:
# python driver.py --mode single --model vgg16 --batch_size 64 --learning_rate 0.001

# Single ResNet18 training:
# python driver.py --mode single --model resnet18 --batch_size 32 --learning_rate 0.0005