import matplotlib.pyplot as plt

# plotting.py - updated with richer final report visualizations

def plot_metrics(epoch_times, val_accuracies, quantized_accuracies=None, title="Model"):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epoch_times, marker='o')
    plt.title(f"{title} - Avg Epoch Time (s)")
    plt.xlabel("Epoch")
    plt.ylabel("Time (s)")

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, marker='o', label="Validation Accuracy")
    if quantized_accuracies:
        plt.plot(quantized_accuracies, marker='x', label="Quantized Accuracy")
    plt.title(f"{title} - Validation Accuracy (%)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_comparison(default_result, tuned_result):
    models = ["Default", "Tuned"]
    accuracies = [default_result["accuracy"], tuned_result["accuracy"]]
    times = [default_result["avg_epoch_time"], tuned_result["avg_epoch_time"]]
    latency = [default_result.get("inference_latency", 0), tuned_result.get("inference_latency", 0)]
    memory = [default_result.get("peak_memory_MB", 0), tuned_result.get("peak_memory_MB", 0)]

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    axs[0, 0].bar(models, accuracies, color=['blue', 'green'])
    axs[0, 0].set_title("Final Validation Accuracy (%)")
    axs[0, 0].set_ylim(0, 100)

    axs[0, 1].bar(models, times, color=['blue', 'green'])
    axs[0, 1].set_title("Average Epoch Time (s)")

    axs[1, 0].bar(models, latency, color=['blue', 'green'])
    axs[1, 0].set_title("Inference Latency (s)")

    axs[1, 1].bar(models, memory, color=['blue', 'green'])
    axs[1, 1].set_title("Peak Memory Usage (MB)")

    plt.suptitle("Performance Comparison: Default vs Tuned", fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_benchmark_results(results):
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

    axs[0, 1].plot(batch_sizes, accs, marker='o', label='Accuracy')
    if any(qaccs):
        axs[0, 1].plot(batch_sizes, qaccs, marker='x', label='Quantized Accuracy')
    axs[0, 1].set_title("Batch Size vs Accuracy")
    axs[0, 1].set_xlabel("Batch Size")
    axs[0, 1].set_ylabel("Accuracy (%)")
    axs[0, 1].legend()

    axs[1, 0].plot(batch_sizes, latency, marker='o')
    axs[1, 0].set_title("Batch Size vs Inference Latency")
    axs[1, 0].set_xlabel("Batch Size")
    axs[1, 0].set_ylabel("Latency (s)")

    axs[1, 1].plot(batch_sizes, memory, marker='o')
    axs[1, 1].set_title("Batch Size vs Peak Memory Usage")
    axs[1, 1].set_xlabel("Batch Size")
    axs[1, 1].set_ylabel("Memory (MB)")

    plt.suptitle("Batch Size Benchmark Results", fontsize=16)
    plt.tight_layout()
    plt.show()
