import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from utils import extract_frames


# ----------------------
# Identity layer for feature extraction
# ----------------------
class Identity(torch.nn.Module):
    def forward(self, x):
        return x


def get_resnet101_feature_extractor(device):
    resnet101 = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    resnet101.fc = Identity()  # remove classifier head
    resnet101 = resnet101.to(device)
    resnet101.eval()
    return resnet101


# ----------------------
# Preprocessing for frames
# ----------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ----------------------
# Main feature extraction function
# ----------------------
def process_and_save_features(video_root, save_root, classes, num_frames, model, device):
    os.makedirs(save_root, exist_ok=True)

    for class_name in classes:
        class_dir = os.path.join(video_root, class_name)
        save_class_dir = os.path.join(save_root, class_name)
        os.makedirs(save_class_dir, exist_ok=True)

        video_files = [f for f in os.listdir(class_dir) if f.endswith('.avi')]

        for video_file in tqdm(video_files, desc=f"Processing {class_name}"):
            video_path = os.path.join(class_dir, video_file)

            try:
                frames = extract_frames(video_path, num_frames)
            except Exception as e:
                print(f"Skipping {video_path}: {e}")
                continue

            input_tensors = torch.stack([transform(frame) for frame in frames]).to(device)

            with torch.no_grad():
                features = model(input_tensors)  # (T, 2048)

            save_path = os.path.join(save_class_dir, video_file.replace('.avi', '.pt'))
            torch.save(features.cpu(), save_path)


# ----------------------
# Entry point
# ----------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", type=str, default="data/UCF50",
                        help="Path to raw UCF50 videos")
    parser.add_argument("--save_root", type=str, default="output/ucf50_features",
                        help="Path to save extracted features")
    parser.add_argument("--classes_file", type=str, default="data/classes.txt",
                        help="File containing list of class names")
    parser.add_argument("--num_frames", type=int, default=24,
                        help="Number of frames to sample per video")
    args = parser.parse_args()

    # Load class names
    with open(args.classes_file, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_resnet101_feature_extractor(device)

    process_and_save_features(args.video_root, args.save_root, classes,
                              args.num_frames, model, device)

    print(f"\n Feature extraction completed. Saved to {args.save_root}/")

