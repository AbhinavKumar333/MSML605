import matplotlib.pyplot as plt
import pandas as pd

# ========================================
# Plot: Per-Epoch Training Metrics
# ========================================

def plot_training_metrics(epoch_times, val_accuracies, quantized_accuracies, title="Model"):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epoch_times, marker='o')
    plt.title(f"{title} - Avg Epoch Time (s)")
    plt.xlabel("Batch Size Index")
    plt.ylabel("Time (s)")

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, marker='o', label="Normal Accuracy")
    plt.plot(quantized_accuracies, marker='x', label="Quantized Accuracy")
    plt.title(f"{title} - Validation Accuracies (%)")
    plt.xlabel("Batch Size Index")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.show()

# ========================================
# Plot: Default vs Tuned Comparison
# ========================================

def plot_default_vs_tuned_comparison(default_result, tuned_result):
    models = ["Default", "Tuned"]
    accuracies = [default_result["accuracy"], tuned_result["accuracy"]]
    times = [default_result["avg_epoch_time"], tuned_result["avg_epoch_time"]]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy Plot
    axs[0].bar(models, accuracies, color=['blue', 'green'])
    axs[0].set_title("Final Validation Accuracy (%)")
    axs[0].set_ylim(0, 100)

    # Epoch Time Plot
    axs[1].bar(models, times, color=['blue', 'green'])
    axs[1].set_title("Average Epoch Time (seconds)")

    plt.suptitle("Performance Comparison: Default vs Tuned", fontsize=16)
    plt.tight_layout()
    plt.show()

# ========================================
# Plot: Batch Size Sweep Results (Multiple Models)
# ========================================

def plot_batchsize_sweep(results, title="Batch Size vs Performance (CPU/GPU)"):
    df = pd.DataFrame(results)
    models = df['model'].unique()

    plt.figure(figsize=(14, 6))

    # Plot Average Epoch Time
    plt.subplot(1, 2, 1)
    for model in models:
        subset = df[df['model'] == model]
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
        subset = df[df['model'] == model]
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
    qaccs = [r['quantized_accuracy'] for r in results]

    plt.figure(figsize=(10, 4))

    # Plot 1: Batch Size vs Epoch Time
    plt.subplot(1, 2, 1)
    plt.plot(batch_sizes, times, marker='o', color='blue')
    plt.title("Batch Size vs Epoch Time")
    plt.xlabel("Batch Size")
    plt.ylabel("Avg Time (s)")

    # Plot 2: Batch Size vs Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(batch_sizes, accs, marker='o', label='Accuracy')
    plt.plot(batch_sizes, qaccs, marker='x', label='Quantized Accuracy')
    plt.title("Batch Size vs Accuracy")
    plt.xlabel("Batch Size")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.show()
