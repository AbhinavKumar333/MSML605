import torch
import torch.nn as nn
import torch.optim as optim
from models.convolution_neural_network import VGG16Modified
from data.loader import get_cifar10_loaders
from evaluation.evaluate import evaluate
from utils.plotting import plot_metrics
from training.loop import train
from config.defaults import config

import numpy as np
import os
from memory_profiler import memory_usage

def train_gpu_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device.upper()}")

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=config["batch_size"], data_dir=config["data_dir"]
    )

    model = VGG16Modified(num_classes=config["num_classes"], input_channels=config["input_channels"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    epoch_times = []
    val_accuracies = []

    for epoch in range(config["epochs"]):
        loss, duration = train(model, train_loader, optimizer, criterion, device=device)
        epoch_times.append(duration)
        print(f"Epoch {epoch+1}/{config['epochs']} - Loss: {loss:.4f} - Time: {duration:.2f}s")

        acc = evaluate(model, test_loader, device=device)
        val_accuracies.append(acc)

    print("\nTraining complete.")
    print(f"Avg epoch time: {np.mean(epoch_times):.2f}s")
    print(f"Final validation accuracy: {val_accuracies[-1]:.2f}%")

    plot_metrics(epoch_times, val_accuracies)

if __name__ == "__main__":
    mem = memory_usage((train_gpu_model,), max_iterations=1)
    print(f"Peak memory usage: {max(mem):.2f} MB")
