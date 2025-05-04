import matplotlib.pyplot as plt

# printing.py - enhanced to complement plotting.py

def print_benchmark_table(results):
    from collections import defaultdict

    grouped = defaultdict(list)
    for r in results:
        grouped[r["model_variant"]].append(r)

    for model_variant, model_results in grouped.items():
        print(f"\n=== Benchmark: {model_variant.upper()} ===")
        print("{:<12} {:<18} {:<18} {:<18} {:<18}".format(
            "Batch Size", "Epoch Time (s)", "Accuracy (%)", "Quantized Acc (%)", "Peak Memory (MB)"
        ))
        print("-" * 85)
        for r in model_results:
            print("{:<12} {:<18.2f} {:<18.2f} {:<18} {:<18}".format(
                r['batch_size'],
                r['avg_epoch_time'],
                r['accuracy'],
                f"{r['quantized_accuracy']:.2f}" if r.get('quantized_accuracy') else "N/A",
                f"{r['peak_memory_MB']:.2f}" if r.get('peak_memory_MB') else "N/A"
            ))

def print_single_result(result):
    print("\n🔎 Final Result Summary:")
    print(f"Batch Size       : {result.get('batch_size', 'N/A')}")
    print(f"Avg Epoch Time   : {result.get('avg_epoch_time', 0):.2f} sec")
    print(f"Final Accuracy   : {result.get('accuracy', 0):.2f}%")
    print(f"Inference Latency: {result.get('inference_latency', 0):.4f} sec")
    print(f"Peak Memory Usage: {result.get('peak_memory_MB', 0):.2f} MB")
    if 'quantized_accuracy' in result:
        print(f"Quantized Accuracy: {result['quantized_accuracy']:.2f}%")
