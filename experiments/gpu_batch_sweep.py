from Features.gpu_optimized import train_gpu_model
from Utils.plotting import plot_batchsize_sweep

def batch_size_sweep_gpu():
    batch_sizes = [16, 32, 64, 128]
    model_variants = ["resnet18", "mobilenetv2"]
    all_results = []

    print("=== Starting GPU Batch Size Sweep ===")

    for model_variant in model_variants:
        print(f"\n Benchmarking {model_variant.upper()} on GPU...")
        for batch_size in batch_sizes:
            print(f"Batch Size: {batch_size}")
            result = train_gpu_model(
                batch_size=batch_size,
                model_variant=model_variant,
                epochs=10,
                learning_rate=0.0005,
                subset=False,
                verbose=True
            )
            result.update({
                "model": model_variant,
                "batch_size": batch_size
            })
            all_results.append(result)

    # Plot all results at once (non-blocking)
    plot_batchsize_sweep(all_results, title="GPU Optimization Sweep")

    return all_results
