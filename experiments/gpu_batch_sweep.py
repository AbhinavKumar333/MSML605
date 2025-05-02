from Features.gpu_optimized import train_gpu_model
from Utils.plotting import plot_batchsize_sweep


def batch_size_sweep_gpu(subset=False, dataset_size=5000, epochs=10, learning_rate=0.0005):
    batch_sizes = [16, 32, 64, 128]
    models = ["resnet18", "mobilenetv2"]
    results = []

    for model_variant in models:
        print(f"\n🚀 Benchmarking {model_variant} on GPU...")
        for batch_size in batch_sizes:
            result = train_gpu_model( subset=subset, dataset_size=dataset_size,
                batch_size=batch_size, model_variant=model_variant,
                epochs=epochs, learning_rate=learning_rate, verbose=True
            )
            result["model"] = model_variant
            result["batch_size"] = batch_size  
            results.append(result)
    print(results)
    plot_batchsize_sweep(results, title="GPU Optimization Sweep")

    return result