import os
import torch
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    """
    Dataset that loads pre-extracted ResNet features (.pt files).
    """
    def __init__(self, root_dir, classes):
        self.classes = classes
        self.samples = []

        for label, class_name in enumerate(classes):
            class_dir = os.path.join(root_dir, class_name)
            for fname in os.listdir(class_dir):
                if fname.endswith('.pt'):
                    path = os.path.join(class_dir, fname)
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feature_path, label = self.samples[idx]
        features = torch.load(feature_path)  # shape: (num_frames, 2048)
        return features, label
