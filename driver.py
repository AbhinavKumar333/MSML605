import argparse
from features.cpu_optimized import train_cpu_model
from experiments.cpu_batch_sweep import batch_size_sweep


def main():
    parser = argparse.ArgumentParser(description="Run CPU training or benchmarking.")
    
    parser.add_argument(
        "--mode",
        choices=["single", "sweep"],
        default="single",
        help="Run a single optimized training or batch size sweep."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size to use (for single run only)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs (for single run only)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (for single run only)"
    )

    args = parser.parse_args()

    if args.mode == "single":
        result = train_cpu_model(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            verbose=True
        )
        print("\n✅ Training Complete:")
        print(f"Batch Size: {result['batch_size']}")
        print(f"Avg Epoch Time: {result['avg_epoch_time']:.2f}s")
        print(f"Final Accuracy: {result['accuracy']:.2f}%")
        print(f"Quantized Accuracy: {result['quantized_accuracy']:.2f}%")

    elif args.mode == "sweep":
        batch_size_sweep()


if __name__ == "__main__":
    main()
