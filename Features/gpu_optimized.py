import torch
import torch.nn as nn
import torch.optim as optim
from Models.simplecnn import SimpleCNN
from Models.vgg import VGG16Modified
from Models.resnet import ResNet18Modified
from Models.mobilenet import MobileNetV2Modified
from data.loader import get_cifar10_loaders
from Evaluation.benchmark import train_and_benchmark
from training.loop import train


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


def train_gpu_model(
    subset=False,
    dataset_size=5000,
    batch_size=64,
    model_variant="resnet18",
    epochs=10,
    learning_rate=0.001,
    verbose=True,
    amp=True,
    quantize=False
):

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Running on {device.upper()}")
    print(f"\n[Model Variant: {model_variant}]")
    print(f"\n[Batch Size: {batch_size}]")

    # Load CIFAR-10 dataset
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=batch_size,
        resize_for_vgg=(model_variant.lower() == "vgg16"),
        subset=subset,
        dataset_size=dataset_size
    )

    model_args = {
        "num_classes": 10,
        "input_channels": 3,
        "pretrained": subset
    }
    model = build_model(model_variant, model_args).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Benchmark training
    stats = train_and_benchmark(
        model, train_loader, test_loader,
        optimizer, criterion, device=device,
        epochs=epochs, verbose=verbose,
        trainer_fn=train,
        quantize=quantize,
        use_tracemalloc=(device == "cpu"),
        use_amp=amp
    )

    return {
        "batch_size": batch_size,
        "avg_epoch_time": stats["avg_epoch_time"],
        "accuracy": stats["final_acc"],
        "quantized_accuracy": None,
        "peak_memory_MB": stats.get("peak_memory_MB"),
        "inference_latency": stats["inference_latency"]
    }


if __name__ == "__main__":
    result = train_gpu_model()
    print(result)
