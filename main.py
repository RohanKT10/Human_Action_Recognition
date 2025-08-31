import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import VideoDataset
from utils import create_split_dirs, top_k_accuracy, classification_metrics
from models import BiLSTMAttentionClassifier, TCNClassifierWithAttention
from train import train
from validate import evaluate


# --------------------------
# Reproducibility
# --------------------------
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if using multi-GPU

# deterministic settings
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --------------------------------------------------
    # Dataset preparation
    # --------------------------------------------------
    print("Preparing dataset splits...")

    # Load classes from classes.txt to ensure consistent ordering
    with open("data/classes.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    train_dir, val_dir,_= create_split_dirs(args.feature_root, args.output_root)

    train_dataset = VideoDataset(train_dir, classes)
    val_dataset   = VideoDataset(val_dir, classes)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # --------------------------------------------------
    # Model selection
    # --------------------------------------------------
    if args.model.lower() == "bilstm":
        model = BiLSTMAttentionClassifier(
            input_dim=args.input_dim,
            num_classes=len(classes),
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    elif args.model.lower() == "tcn":
        model = TCNClassifierWithAttention(
            input_dim=args.input_dim,
            num_classes=len(classes),
            dropout=args.dropout
        )
    else:
        raise ValueError("Invalid model type. Choose either 'bilstm' or 'tcn'.")

    model = model.to(device)
    print(f"Model: {args.model}")

    # --------------------------------------------------
    # Training setup
    # --------------------------------------------------
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    best_val_acc = 0.0
    best_epoch = -1

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------

    checkpoint_path = os.path.join(args.output_root, f"checkpoints/best_model_{args.model.lower()}.pth")
    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        model.load_state_dict(torch.load(checkpoint_path))

    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{args.epochs} "
              f"Train Loss={train_loss:.4f} "
              f"Train Acc={train_acc:.4f} "
              f"Val Acc={val_acc:.4f}")

        # Save best model (by top1 acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            os.makedirs(os.path.join(args.output_root, "checkpoints"), exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_root, f"checkpoints/best_model_{args.model.lower()}.pth"))
            print(f" Best model saved (epoch {best_epoch}, Val_acc={best_val_acc:.2f}%)")

        scheduler.step(val_acc)

    print(f"Training completed. Best Val Acc={best_val_acc:.2f}% at epoch {best_epoch}")

    # --------------------------------------------------
    # Final evaluation (Top1, Top5, classification report)
    # --------------------------------------------------
    print("\nEvaluating final model on validation set...")
    model.load_state_dict(torch.load(os.path.join(args.output_root, f"checkpoints/best_model_{args.model.lower()}.pth")))
    model.eval()

    total, top5_correct, top1_correct = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for videos, labels in val_loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            total += labels.size(0)
            top5_correct += top_k_accuracy(outputs, labels, k=5)
            top1_correct += (outputs.argmax(1) == labels).sum().item()
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    top1_acc = 100.0 * top1_correct / total
    top5_acc = 100.0 * top5_correct / total
    print(f"\nFinal Results on Validation Set:\nTop-1 Accuracy: {top1_acc:.2f}%\nTop-5 Accuracy: {top5_acc:.2f}%")

    # Classification report + confusion matrix
    classification_metrics(all_labels, all_preds, classes,
                           save_path=os.path.join(args.output_root, f"confusion_matrix_{args.model.lower()}.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Human Action Recognition with BiLSTM/TCN + Attention")

    # Dataset paths
    parser.add_argument("--feature_root", type=str, required=True,
                        help="Path to pre-extracted features (organized per class)")
    parser.add_argument("--output_root", type=str, default="./outputs",
                        help="Path to save splits, logs, models")

    # Model selection
    parser.add_argument("--model", type=str, choices=["bilstm", "tcn"], default="tcn",
                        help="Which model to use: bilstm or tcn")

    # Model params
    parser.add_argument("--input_dim", type=int, default=2048)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.7)

    # Training params
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()
    main(args)