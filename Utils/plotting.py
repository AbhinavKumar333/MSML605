import matplotlib.pyplot as plt

def plot_metrics(epoch_times, val_accuracies):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_times, marker='o')
    plt.title("Epoch Time (seconds)")
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, marker='o')
    plt.title("Validation Accuracy (%)")
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
