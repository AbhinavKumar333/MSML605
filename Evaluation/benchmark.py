import time
import numpy as np
import torch
from evaluation.evaluate import evaluate

def train_and_benchmark(model, train_loader, test_loader, optimizer, criterion, device, epochs=2, verbose=True, trainer_fn=None):
    epoch_times = []
    val_accuracies = []

    for epoch in range(epochs):
        start = time.perf_counter()
        loss, _ = trainer_fn(model, train_loader, optimizer, criterion, device=device)
        duration = time.perf_counter() - start
        epoch_times.append(duration)

        acc = evaluate(model, test_loader, device=device)
        val_accuracies.append(acc)

        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} - Time: {duration:.2f}s - Acc: {acc:.2f}%")

    return {
        "epoch_times": epoch_times,
        "val_accuracies": val_accuracies,
        "final_acc": val_accuracies[-1],
        "avg_epoch_time": np.mean(epoch_times),
    }

def benchmark_quantized(model, test_loader, device="cpu"):
    quant_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    acc_quant = evaluate(quant_model, test_loader, device=device)
    return acc_quant
