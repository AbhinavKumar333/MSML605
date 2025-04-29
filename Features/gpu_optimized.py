import torch
import torch.nn as nn
import torch.optim as optim
from Models.convolution_neural_network import VGG16Modified
from data.loader import get_cifar10_loaders
from Evaluation.evaluate import evaluate
from Utils.plotting import plot_metrics
from training.loop import train
from config.defaults import config
# from Models.convolution_neural_network import VGG16Modified
from Models.resnet import ResNet18Modified
from Models.mobilenet import MobileNetV2Modified

import numpy as np
import os
from memory_profiler import memory_usage

# def train_gpu_model(batch_size=64, model_variant="vgg16", epochs=10, learning_rate=0.001, verbose=False, subset=False):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Running on {device.upper()}")

#     train_loader, test_loader = get_cifar10_loaders(
#         batch_size=config["batch_size"], data_dir=config["data_dir"]
#     )

#     model = VGG16Modified(num_classes=config["num_classes"], input_channels=config["input_channels"]).to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

#     epoch_times = []
#     val_accuracies = []

#     for epoch in range(config["epochs"]):
#         loss, duration = train(model, train_loader, optimizer, criterion, device=device)
#         epoch_times.append(duration)
#         print(f"Epoch {epoch+1}/{config['epochs']} - Loss: {loss:.4f} - Time: {duration:.2f}s")

#         acc = evaluate(model, test_loader, device=device)
#         val_accuracies.append(acc)

#     print("\nTraining complete.")
#     print(f"Avg epoch time: {np.mean(epoch_times):.2f}s")
#     print(f"Final validation accuracy: {val_accuracies[-1]:.2f}%")

#     plot_metrics(epoch_times, val_accuracies)


def train_gpu_model(batch_size=64, model_variant="vgg16", epochs=10, learning_rate=0.001, verbose=False, subset=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device.upper()}")

    # Use batch_size and subset passed into the function
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=batch_size,
        data_dir=config["data_dir"],  # data_dir can still come from config
        subset=subset
    )

    # Dynamically select model based on model_variant
    model = build_model(model_variant).to(device)

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

    # Optionally print final results
    if verbose:
        print("\nTraining complete.")
        print(f"Avg epoch time: {np.mean(epoch_times):.2f}s")
        print(f"Final validation accuracy: {val_accuracies[-1]:.2f}%")

    # Return a dictionary of results (no need to plot inside this function)
    return {
        "avg_epoch_time": np.mean(epoch_times),
        "accuracy": val_accuracies[-1],
        "epoch_times": epoch_times,
        "val_accuracies": val_accuracies
    }





def build_model(model_variant):
    if model_variant == "vgg16":
        return VGG16Modified(num_classes=10, input_channels=3)
    elif model_variant == "resnet18":
        return ResNet18Modified(num_classes=10, input_channels=3)
    elif model_variant == "mobilenetv2":
        return MobileNetV2Modified(num_classes=10, input_channels=3)
    else:
        raise ValueError(f"Unknown model variant: {model_variant}")


if __name__ == "__main__":
    mem = memory_usage((train_gpu_model,), max_iterations=1)
    print(f"Peak memory usage: {max(mem):.2f} MB")
