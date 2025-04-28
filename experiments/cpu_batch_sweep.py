from features.cpu_optimized import train_cpu_model
from utils.printing import print_benchmark_table
from utils.plotting import plot_benchmark_results, plot_metrics

def batch_size_sweep():
    # batch_sizes = [8, 16, 32, 64, 128]
    batch_sizes = [8, 32]  # Reduced for faster testing
    model_variants = ["vgg16", "resnet18", "mobilenetv2"]

    sweep_results = []

    print("📊 Starting CPU batch size sweep across models...\n")

    for model_variant in model_variants:
        print(f"\nBenchmarking Model: {model_variant.upper()}\n")
        
        for bs in batch_sizes:
            print(f"Batch Size: {bs}")

            # Normal training
            result_normal = train_cpu_model(batch_size=bs, verbose=False, model_variant=model_variant, quantize=False)

            # Quantized evaluation
            result_quantized = train_cpu_model(batch_size=bs, verbose=False, model_variant=model_variant, quantize=True)

            # Record
            sweep_results.append({
                "model_variant": model_variant,
                "batch_size": bs,
                "avg_epoch_time": result_normal["avg_epoch_time"],
                "accuracy": result_normal["accuracy"],
                "quantized_accuracy": result_quantized["accuracy"]
            })

    print_benchmark_table(sweep_results)

    for model in model_variants:
        model_data = [r for r in sweep_results if r["model_variant"] == model]
        epoch_times = [r["avg_epoch_time"] for r in model_data]
        val_accs = [r["accuracy"] for r in model_data]
        quant_accs = [r["quantized_accuracy"] for r in model_data]

        plot_metrics(epoch_times, val_accs, quant_accs, title=model.upper())


if __name__ == "__main__":
    batch_size_sweep()
