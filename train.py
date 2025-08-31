import torch
from tqdm import tqdm

def train(model, dataloader, optimizer, criterion, device):
    """
    Training loop for one epoch (full precision, no AMP).
    """
    model.train()
    total_loss = 0.0
    correct = 0

    for frames, labels in tqdm(dataloader, desc="Training"):
        frames, labels = frames.to(device), labels.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(frames)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        #clip gradients to stabilize training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)

    return avg_loss, accuracy
