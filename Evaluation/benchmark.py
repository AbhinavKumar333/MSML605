import time
import numpy as np
import torch
import psutil
import os
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
    quantize=False,
    use_tracemalloc=False,
    use_amp=False
):
    epoch_times = []
    val_accuracies = []
    peak_memory_MB = 0

    is_cuda = device == "cuda"
    is_mps = device == "mps"
    is_cpu = device == "cpu"

    if use_tracemalloc and is_cpu:
        tracemalloc.start()

    if is_cuda:
        torch.cuda.reset_peak_memory_stats()

    scaler = torch.cuda.amp.GradScaler() if use_amp and is_cuda else None

    for epoch in range(epochs):
        if is_cuda:
            torch.cuda.reset_peak_memory_stats()

        start_time = time.perf_counter()

        # === Training ===
        if use_amp and is_cuda:
            model.train()
            total_loss = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
            loss = total_loss / len(train_loader)
        else:
            loss, _ = trainer_fn(model, train_loader, optimizer, criterion, device=device)

        duration = time.perf_counter() - start_time
        epoch_times.append(duration)

        # === Evaluation ===
        acc = evaluate(model, test_loader, device=device)
        val_accuracies.append(acc)

        # === Memory Tracking ===
        if is_cuda:
            mem_bytes = torch.cuda.max_memory_allocated()
        elif use_tracemalloc and is_cpu:
            current, peak = tracemalloc.get_traced_memory()
            mem_bytes = peak
        else:
            process = psutil.Process(os.getpid())
            mem_bytes = process.memory_info().rss

        mem_mb = mem_bytes / (1024 * 1024)
        peak_memory_MB = max(peak_memory_MB, mem_mb)

        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - Time: {duration:.2f}s - Acc: {acc:.2f}% - Peak Mem: {mem_mb:.2f} MB")

    if use_tracemalloc and is_cpu:
        tracemalloc.stop()

    # === Quantization ===
    final_model = model
    if quantize and is_cpu:
        if verbose:
            print("Applying dynamic quantization for final evaluation...")
        final_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    # === Final Evaluation and Inference Latency ===
    final_acc = evaluate(final_model, test_loader, device=device)
    start_inf = time.perf_counter()
    _ = evaluate(final_model, test_loader, device=device)
    inference_latency = time.perf_counter() - start_inf

    return {
        "epoch_times": epoch_times,
        "val_accuracies": val_accuracies,
        "final_acc": final_acc,
        "avg_epoch_time": np.mean(epoch_times),
        "inference_latency": inference_latency,
        "peak_memory_MB": peak_memory_MB
    }
