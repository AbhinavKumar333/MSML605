import matplotlib.pyplot as plt


def plot_metrics(epoch_times, val_accuracies, quantized_accuracies, title="Model"):
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

def plot_comparison(default_result, tuned_result):
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

def plot_benchmark_results(results):
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
