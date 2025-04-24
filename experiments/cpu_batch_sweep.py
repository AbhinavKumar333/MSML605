from features.cpu_optimized import train_cpu_model
from utils.printing import print_benchmark_table
from utils.plotting import plot_benchmark_results

def batch_size_sweep():
    batch_sizes = [16, 32, 64]
    results = []

    print("\n📊 Starting CPU batch size sweep...\n")
    for bs in batch_sizes:
        result = train_cpu_model(batch_size=bs, verbose=False)
        results.append(result)

    print_benchmark_table(results)
    plot_benchmark_results(results)
    return results


if __name__ == "__main__":
    batch_size_sweep()
