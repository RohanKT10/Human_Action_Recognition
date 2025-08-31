import torch
from tqdm import tqdm

def evaluate(model, dataloader, device):
    """
    Evaluation loop — returns accuracy (%).
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for videos, labels in tqdm(dataloader, desc="Validating"):
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total * 100
    return acc
