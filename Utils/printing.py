import matplotlib.pyplot as plt

def print_benchmark_table(results):
    batch_sizes = [r['batch_size'] for r in results]
    times = [r['avg_epoch_time'] for r in results]
    accs = [r['accuracy'] for r in results]
    qaccs = [r['quantized_accuracy'] for r in results]

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(batch_sizes, times, marker='o')
    plt.title("Batch Size vs Epoch Time")
    plt.xlabel("Batch Size")
    plt.ylabel("Time (s)")

    plt.subplot(1, 2, 2)
    plt.plot(batch_sizes, accs, label='Accuracy', marker='o')
    plt.plot(batch_sizes, qaccs, label='Quantized Accuracy', marker='x')
    plt.title("Accuracy vs Batch Size")
    plt.xlabel("Batch Size")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.show()
