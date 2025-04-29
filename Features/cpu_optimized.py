import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from Models import convolution_neural_network, resnet, mobilenet
from data.loader import get_cifar10_loaders
from Evaluation.evaluate import evaluate
from training.loop import train
from Evaluation.benchmark import train_and_benchmark, benchmark_quantized


from Models import convolution_neural_network, resnet, mobilenet

def train_cpu_model(batch_size=32, epochs=2, learning_rate=0.001, verbose=True, model_variant="vgg16", quantize=False):
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
    torch.set_num_threads(os.cpu_count())

    if verbose:
        print(f"\n[Batch Size: {batch_size}]")
        print(f"Running on CPU with {torch.get_num_threads()} threads")
        print("MKL Enabled in PyTorch:", torch.backends.mkl.is_available())

    # Load CIFAR-10 (TEST)
    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size, subset=True)

    # === Select model based on variant ===
    if model_variant.lower() == "vgg16":
        model = convolution_neural_network.VGG16Modified(num_classes=10, input_channels=3)
    elif model_variant.lower() == "resnet18":
        model = resnet.get_model(num_classes=10)
    elif model_variant.lower() == "mobilenetv2":
        model = mobilenet.get_model(num_classes=10)
    else:
        raise ValueError(f"Unsupported model variant: {model_variant}")

    model = model.to("cpu")

    # Optional optimization with torch.compile
    try:
        # model = torch.compile(model)
        if verbose:
            print("Model compiled with torch.compile()")
    except Exception:
        if verbose:
            print("torch.compile() not available, using regular model")

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training and Benchmarking
    stats = train_and_benchmark(
        model, train_loader, test_loader, optimizer, criterion,
        device="cpu", epochs=epochs, verbose=verbose, trainer_fn=train,
        quantize=False  # or True if you want quantized final evaluation
    )

    # Quantization Benchmark
    if quantize:
        acc_quant = benchmark_quantized(model, test_loader, device="cpu")
        if verbose:
            print(f"Quantized Accuracy: {acc_quant:.2f}%")
    else:
        acc_quant = None

    return {
        "batch_size": batch_size,
        "avg_epoch_time": stats["avg_epoch_time"],
        "accuracy": stats["final_acc"],
        "quantized_accuracy": acc_quant
    }



if __name__ == "__main__":
    train_cpu_model()
