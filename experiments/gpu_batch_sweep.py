from features.gpu_optimized import train_gpu_model
from utils.plotting import plot_batchsize_vs_metrics

def batch_size_sweep_gpu():
    batch_sizes = [16, 32, 64, 128]
    models = ["vgg16", "resnet18", "mobilenetv2"]
    results = []

    for model_variant in models:
        print(f"\n🚀 Benchmarking {model_variant} on GPU...")
        for batch_size in batch_sizes:
            result = train_gpu_model(
                batch_size=batch_size, model_variant=model_variant,
                epochs=2, learning_rate=0.001, verbose=True, subset=True
            )
            result["model"] = model_variant
            results.append(result)

    plot_batchsize_vs_metrics(results, title="GPU Optimization Sweep")