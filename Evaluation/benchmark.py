import time
import numpy as np
import torch
import tracemalloc
from Evaluation.evaluate import evaluate

def train_and_benchmark(
    model,
    train_loader,
    test_loader,
    optimizer,
    criterion,
    device,
    epochs=2,
    verbose=True,
    trainer_fn=None,
    quantize=False
):
    epoch_times = []
    val_accuracies = []
    max_memory = 0

    tracemalloc.start()

    for epoch in range(epochs):
        start = time.perf_counter()
        loss, _ = trainer_fn(model, train_loader, optimizer, criterion, device=device)
        duration = time.perf_counter() - start
        epoch_times.append(duration)

        acc = evaluate(model, test_loader, device=device)
        val_accuracies.append(acc)

        current, peak = tracemalloc.get_traced_memory()
        max_memory = max(max_memory, peak / (1024 * 1024))  # MB

        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} - Time: {duration:.2f}s - Acc: {acc:.2f}%")

    final_model = model
    if quantize:
        if verbose:
            print("Applying dynamic quantization for final evaluation...")
        final_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    # Final accuracy (with or without quantization)
    final_acc = evaluate(final_model, test_loader, device=device)

    # Measure inference latency
    start_inf = time.perf_counter()
    _ = evaluate(final_model, test_loader, device=device)
    inference_latency = time.perf_counter() - start_inf

    tracemalloc.stop()

    return {
        "epoch_times": epoch_times,
        "val_accuracies": val_accuracies,
        "final_acc": final_acc,
        "avg_epoch_time": np.mean(epoch_times),
        "inference_latency": inference_latency,
        "peak_memory_MB": max_memory
    }


def benchmark_quantized(model, test_loader, device="cpu"):
    quant_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    acc_quant = evaluate(quant_model, test_loader, device=device)
    return acc_quant
