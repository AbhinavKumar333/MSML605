import matplotlib.pyplot as plt
import pandas as pd
import optuna

# ========================================
# Plot: Per-Epoch Training Metrics
# ========================================
def plot_metrics(epoch_times, val_accuracies, quantized_accuracies=None, title="Model"):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epoch_times, marker='o')
    plt.title(f"{title} - Avg Epoch Time (s)")
    plt.xlabel("Epoch")
    plt.ylabel("Time (s)")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, marker='o', label="Validation Accuracy")
    if quantized_accuracies:
        plt.plot(quantized_accuracies, marker='x', label="Quantized Accuracy")
    plt.title(f"{title} - Validation Accuracy (%)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

# ========================================
# Plot: Default vs Tuned Comparison
# ========================================
def plot_all_comparisons(comparisons):
    metrics = {
        "Accuracy (%)": ("accuracy", "%"),
        "Avg Epoch Time (s)": ("avg_epoch_time", "s"),
        "Inference Latency (s)": ("inference_latency", "s"),
        "Peak Memory Usage (MB)": ("peak_memory_MB", "MB")
    }

    for model_name, runs in comparisons:
        strategies = [label for label, _ in runs]

        for metric_title, (key, unit) in metrics.items():
            values = [result.get(key, 0) for _, result in runs]

            plt.figure(figsize=(10, 5))
            bars = plt.bar(strategies, values, color='skyblue')
            plt.title(f"{metric_title} for {model_name}")
            plt.ylabel(f"{metric_title} ({unit})")
            plt.xticks(rotation=30, ha='right')

            # Annotate bars
            for bar, val in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f"{val:.2f}", ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            plt.show()


# ========================================
# Plot: Batch Size Sweep Results (Multiple Models)
# ========================================
def plot_batchsize_sweep(results, title="Batch Size vs Performance (CPU/GPU)"):
    df = pd.DataFrame(results)
    models = df['model_variant'].unique()  # Adjust to match your result dict key

    plt.figure(figsize=(14, 6))

    # Plot Average Epoch Time
    plt.subplot(1, 2, 1)
    for model in models:
        subset = df[df['model_variant'] == model]
        plt.plot(subset['batch_size'], subset['avg_epoch_time'], marker='o', label=model)
    plt.xlabel("Batch Size")
    plt.ylabel("Average Epoch Time (seconds)")
    plt.title("Epoch Time vs Batch Size")
    plt.xscale('log', base=2)
    plt.grid(True)
    plt.legend()

    # Plot Validation Accuracy
    plt.subplot(1, 2, 2)
    for model in models:
        subset = df[df['model_variant'] == model]
        plt.plot(subset['batch_size'], subset['accuracy'], marker='o', label=model)
    plt.xlabel("Batch Size")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Accuracy vs Batch Size")
    plt.xscale('log', base=2)
    plt.grid(True)
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# ========================================
# Plot: Benchmark Results (Single Model)
# ========================================
def plot_benchmark_single(results):
    batch_sizes = [r['batch_size'] for r in results]
    times = [r['avg_epoch_time'] for r in results]
    accs = [r['accuracy'] for r in results]
    qaccs = [r.get('quantized_accuracy') for r in results]
    latency = [r.get('inference_latency') for r in results]
    memory = [r.get('peak_memory_MB') for r in results]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    axs[0, 0].plot(batch_sizes, times, marker='o')
    axs[0, 0].set_title("Batch Size vs Epoch Time")
    axs[0, 0].set_xlabel("Batch Size")
    axs[0, 0].set_ylabel("Avg Time (s)")
    axs[0, 0].grid(True)

    axs[0, 1].plot(batch_sizes, accs, marker='o', label='Accuracy')
    if any(qaccs):
        axs[0, 1].plot(batch_sizes, qaccs, marker='x', label='Quantized Accuracy')
    axs[0, 1].set_title("Batch Size vs Accuracy")
    axs[0, 1].set_xlabel("Batch Size")
    axs[0, 1].set_ylabel("Accuracy (%)")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    axs[1, 0].plot(batch_sizes, latency, marker='o')
    axs[1, 0].set_title("Batch Size vs Inference Latency")
    axs[1, 0].set_xlabel("Batch Size")
    axs[1, 0].set_ylabel("Latency (s)")
    axs[1, 0].grid(True)

    axs[1, 1].plot(batch_sizes, memory, marker='o')
    axs[1, 1].set_title("Batch Size vs Peak Memory Usage")
    axs[1, 1].set_xlabel("Batch Size")
    axs[1, 1].set_ylabel("Memory (MB)")
    axs[1, 1].grid(True)

    plt.suptitle("Batch Size Benchmark Results", fontsize=16)
    plt.tight_layout()
    plt.show()

# ========================================
# Plot: Optuna Optimization History
# ========================================
def plot_optimization_history(study):
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title("Optuna Optimization History")
    plt.grid(True)
    plt.show()