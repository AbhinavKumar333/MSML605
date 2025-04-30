
import torch
import torch.nn as nn
import torch.optim as optim
from Models.simplecnn import SimpleCNN
from Models.vgg import VGG16Modified
from Data.loader import get_cifar10_loaders
from Evaluation.evaluate import evaluate
from Training.loop import train
from Models.resnet import ResNet18Modified
from Models.mobilenet import MobileNetV2Modified
import numpy as np
import os
from Evaluation.benchmark import benchmark_quantized
from memory_profiler import memory_usage


def build_model(model_variant, model_args):
    if model_variant == "simplecnn":
        return SimpleCNN(**model_args)
    elif model_variant == "vgg16":
        return VGG16Modified(**model_args)
    elif model_variant == "resnet18":
        return ResNet18Modified(**model_args)
    elif model_variant == "mobilenetv2":
        return MobileNetV2Modified(**model_args)
    else:
        raise ValueError(f"Unknown model variant: {model_variant}")


def train_gpu_model(subset=False, dataset_size=5000, batch_size=64, model_variant="vgg16", epochs=10, learning_rate=0.001, verbose=False, quantize=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device.upper()}")

    # Use batch_size and subset passed into the function
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=batch_size,
        data_dir='./Data',
        resize_for_vgg= model_variant.lower() == "vgg16",
        subset=subset,
        dataset_size=dataset_size
    )

    # Dynamically select model based on model_variant
    model_args = {
        "num_classes": 10,
        "input_channels": 3,
        "pretrained": subset == True
    }
    model = build_model(model_variant, model_args).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    epoch_times = []
    val_accuracies = []

    for epoch in range(epochs):
        loss, duration = train(model, train_loader, optimizer, criterion, device=device)
        epoch_times.append(duration)

        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Time: {duration:.2f}s")

        acc = evaluate(model, test_loader, device=device)
        val_accuracies.append(acc)
    
    # Quantization Benchmark
    if quantize:
        acc_quant = benchmark_quantized(model, test_loader, device=device)
        if verbose:
            print(f"Quantized Accuracy: {acc_quant:.2f}%")
    else:
        acc_quant = None

    # Optionally print final results
    if verbose:
        print("\nTraining complete.")
        print(f"Avg epoch time: {np.mean(epoch_times):.2f}s")
        print(f"Final validation accuracy: {val_accuracies[-1]:.2f}%")

    # Return a dictionary of results (no need to plot inside this function)
    return {
        "batch_size": batch_size,
        "avg_epoch_time": np.mean(epoch_times),
        "accuracy": val_accuracies[-1],
        "quantized_accuracy": acc_quant
    }


if __name__ == "__main__":
    mem = memory_usage((train_gpu_model,), max_iterations=1)
    print(f"Peak memory usage: {max(mem):.2f} MB")

