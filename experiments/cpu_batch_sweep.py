
from Features.cpu_optimized import train_cpu_model
from Utils.printing import print_benchmark_table
from Utils.plotting import plot_metrics


def batch_size_sweep(epochs=2, lr=0.001):
    sweep_results = []
    batch_sizes = [8, 16, 32, 64, 128]
    # batch_sizes = [8, 32]  # Reduced for faster testing
    model_variants = ["resnet18", "mobilenetv2"]

    sweep_results = []

    print("Starting CPU batch size sweep across models...\n")

    for model_variant in model_variants:
        print(f"\nBenchmarking Model: {model_variant.upper()}\n")

        for bs in batch_sizes:
            print(f"Batch Size: {bs}")

            # # Normal training
            # result = train_cpu_model(batch_size=bs, epochs=epochs, learning_rate=lr, model_variant=model_variant, quantize=False)

            # Quantized evaluation
            result = train_cpu_model(
                batch_size=bs,
                epochs=epochs,
                learning_rate=lr,
                model_variant=model_variant,
                quantize=True,
                subset=True,
                dataset_size=10000
            )

            # Record
            sweep_results.append({
                "model_variant": model_variant,
                "batch_size": bs,
                "avg_epoch_time": result["avg_epoch_time"],
                "accuracy": result["accuracy"],
                "quantized_accuracy": result["quantized_accuracy"]
            })
    print_benchmark_table(sweep_results)

    for model in model_variants:
        model_data = [r for r in sweep_results if r["model_variant"] == model]
        epoch_times = [r["avg_epoch_time"] for r in model_data]
        val_accs = [r["accuracy"] for r in model_data]
        quant_accs = [r["quantized_accuracy"] for r in model_data]

        plot_metrics(epoch_times, val_accs, quant_accs, title=model.upper())

    return sweep_results

if __name__ == "__main__":
    batch_size_sweep()
