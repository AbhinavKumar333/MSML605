import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from models.convolution_neural_network import VGG16Modified
from data.loader import get_cifar10_loaders
from evaluation.evaluate import evaluate
from training.loop import train
from evaluation.benchmark import train_and_benchmark, benchmark_quantized


def train_cpu_model(batch_size=32, epochs=2, learning_rate=0.001, verbose=True):
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
    torch.set_num_threads(os.cpu_count())

    if verbose:
        print(f"\n[Batch Size: {batch_size}]")
        print(f"Running on CPU with {torch.get_num_threads()} threads")
        print("MKL Enabled in PyTorch:", torch.backends.mkl.is_available())

    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size)

    model = VGG16Modified(num_classes=10, input_channels=3).to("cpu")

    try:
        model = torch.compile(model)
        if verbose:
            print("Model compiled with torch.compile()")
    except Exception:
        if verbose:
            print("torch.compile() not available")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    stats = train_and_benchmark(
        model, train_loader, test_loader, optimizer, criterion,
        device="cpu", epochs=epochs, verbose=verbose, trainer_fn=train
    )

    acc_quant = benchmark_quantized(model, test_loader, device="cpu")

    if verbose:
        print(f"Quantized Accuracy: {acc_quant:.2f}%")

    return {
        "batch_size": batch_size,
        "avg_epoch_time": stats["avg_epoch_time"],
        "accuracy": stats["final_acc"],
        "quantized_accuracy": acc_quant
    }


if __name__ == "__main__":
    train_cpu_model()
