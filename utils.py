import os
import glob
import shutil
import numpy as np
import torch
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------
# Frame extraction
# ----------------------
def extract_frames(video_path, num_frames=24, resize=(224, 224)):
    """Extract uniformly sampled frames from a video."""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if len(frames) < num_frames:
        raise ValueError(f"Video has only {len(frames)} frames, but {num_frames} requested.")

    indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
    selected_frames = [frames[i] for i in indices]

    return np.array(selected_frames)


# ----------------------
# Group ID utilities (UCF50 uses g01–g25 for split)
# ----------------------
def extract_group_id(path: str) -> int:
    """Extract group ID (g01–g25) from filename."""
    fname = os.path.basename(path)
    parts = fname.split("_")
    for p in parts:
        if p.startswith("g") and p[1:].isdigit():
            return int(p[1:])
    raise ValueError(f"Group ID not found in {fname}")


def load_ucf50_feature_paths(root_dir):
    """Return all feature file paths, labels, and class names."""
    classes = sorted(os.listdir(root_dir))
    file_paths, labels = [], []

    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(root_dir, class_name)
        files = glob.glob(os.path.join(class_dir, "*.pt"))
        for f in files:
            file_paths.append(f)
            labels.append(class_idx)

    return file_paths, labels, classes


def create_split_dirs(root_dir, output_dir):
    """
    Split dataset into train (groups 1–22) and test (groups 23–25).
    Copy features to new directories for dataloaders.
    """
    file_paths, labels, classes = load_ucf50_feature_paths(root_dir)

    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for c in classes:
        os.makedirs(os.path.join(train_dir, c), exist_ok=True)
        os.makedirs(os.path.join(test_dir, c), exist_ok=True)

    for f, lab in zip(file_paths, labels):
        g = extract_group_id(f)
        class_name = classes[lab]

        if 1 <= g <= 22:
            shutil.copy(f, os.path.join(train_dir, class_name, os.path.basename(f)))
        elif 23 <= g <= 25:
            shutil.copy(f, os.path.join(test_dir, class_name, os.path.basename(f)))

    return train_dir, test_dir, classes

# ----------------------
# Metrics
# ----------------------
def top_k_accuracy(outputs, labels, k=5):
    """Compute top-k accuracy."""
    _, pred = outputs.topk(k, 1, True, True)  # (B, k)
    correct = pred.eq(labels.view(-1, 1).expand_as(pred))
    return correct.sum().item()

def classification_metrics(all_labels, all_preds, classes, save_path=None):
    """Print and optionally save classification report + confusion matrix."""
    report = classification_report(all_labels, all_preds, target_names=classes, digits=4)
    print("\nClassification Report:\n", report)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm, annot=False, xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
