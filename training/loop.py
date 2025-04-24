import time

def train(model, train_loader, optimizer, criterion, device="cpu"):
    model.train()
    start_time = time.time()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    duration = time.time() - start_time
    return running_loss, duration
