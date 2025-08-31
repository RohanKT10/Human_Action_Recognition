import os
import torch
import argparse
import torchvision.models as models
import torchvision.transforms as transforms
from models import BiLSTMAttentionClassifier, TCNClassifierWithAttention
from extract_frames import extract_frames, Identity
import cv2
import matplotlib.pyplot as plt

# ----------------------
# Argument parser
# ----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--video_path", type=str, required=True, help="Path to the video file (.avi)")
parser.add_argument("--output_root", type=str, default="output", help="Output root directory")
parser.add_argument("--model_type", type=str, default="tcn", choices=["tcn", "bilstm"], help="Model type")
parser.add_argument("--num_frames", type=int, default=24, help="Number of frames to sample from the video")
parser.add_argument("--classes_file", type=str, default="data/classes.txt", help="Path to classes.txt file")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------
# Select model path
# ----------------------
model_path = os.path.join(args.output_root, f"checkpoints/best_model_{args.model_type}.pth")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Train and save the model first!")

print(f"Loading model from: {model_path}")

# ----------------------
# Load class names
# ----------------------
with open(args.classes_file, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# ----------------------
# Preprocessing
# ----------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----------------------
# Feature extractor
# ----------------------
resnet101 = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
resnet101.fc = Identity()
resnet101 = resnet101.to(DEVICE)
resnet101.eval()

# ----------------------
# Extract frames and features
# ----------------------
frames = extract_frames(args.video_path, args.num_frames)
frames_tensor = torch.stack([transform(f) for f in frames]).to(DEVICE)

with torch.no_grad():
    features = resnet101(frames_tensor)  # (T, 2048)
features = features.unsqueeze(0)  # (1, T, 2048)

# ----------------------
# Load classifier
# ----------------------
if args.model_type == "bilstm":
    model = BiLSTMAttentionClassifier(input_dim=2048, num_classes=len(classes))
elif args.model_type == "tcn":
    model = TCNClassifierWithAttention(input_dim=2048, num_classes=len(classes))

model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ----------------------
# Prediction
# ----------------------
with torch.no_grad():
    outputs = model(features)  # (1, num_classes)
    probs = torch.softmax(outputs, dim=1)  # convert logits to probabilities

    # Top-1 prediction
    pred_idx = torch.argmax(probs, dim=1).item()
    pred_class = classes[pred_idx]
    print(f"\nPredicted class: {pred_class}")

    # Top-5 predictions
    top5_prob, top5_idx = torch.topk(probs, 5)
    top5_prob = top5_prob.squeeze(0).cpu().numpy()
    top5_idx = top5_idx.squeeze(0).cpu().numpy()

    print("\nTop-5 Predictions with Probabilities:")
    for i, idx in enumerate(top5_idx):
        print(f"{i+1}. {classes[idx]}: {top5_prob[i]*100:.2f}%")

# ----------------------
# Visual output
# ----------------------
frame = frames[0]  # take the first frame
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))

# Left: Video frame with predicted class
ax[0].imshow(frame_rgb)
ax[0].axis('off')
ax[0].set_title(f"Predicted: {pred_class}", fontsize=14)

# Right: Top-5 probabilities bar chart
top5_classes = [classes[i] for i in top5_idx]
ax[1].barh(top5_classes[::-1], top5_prob[::-1]*100, color='skyblue')
ax[1].set_xlim(0, 100)
ax[1].set_xlabel("Probability (%)")
ax[1].set_title("Top-5 Predictions", fontsize=14)

plt.tight_layout()
plt.savefig(f"output/prediction_visual_{args.model_type}.png")  # save the figure
plt.close()