import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from Models.convolution_neural_network import CNN
import time

# Enable cuDNN autotuner to find the best algorithm for the current configuration.
torch.backends.cudnn.benchmark = True

# Set device to GPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Transformation and data loading with pinned memory.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)

# Benchmark the training loop.
start_time = time.time()
model.train()
for epoch in range(1):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Use non-blocking transfers for performance.
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
end_time = time.time()

print("GPU Training time:", end_time - start_time)
