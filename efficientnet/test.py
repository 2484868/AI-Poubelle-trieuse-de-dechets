import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================
MODEL_PATH = "efficientnet/efficientnet_waste_4class.pt"
TEST_IMAGES_DIR = "efficientnet/cropped_images"
IMAGE_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# LOAD MODEL
# =========================================================
checkpoint = torch.load(MODEL_PATH, map_location=device)

class_names = checkpoint["class_names"]
num_classes = len(class_names)

model = efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(in_features, num_classes)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

print("Loaded model with classes:", class_names)


# =========================================================
# TRANSFORM (same as eval)
# =========================================================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# =========================================================
# PREDICTION FUNCTION
# =========================================================
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)  # shape: (1, C, H, W)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

    pred_idx = probs.argmax(dim=1).item()
    confidence = probs[0, pred_idx].item()

    return class_names[pred_idx], confidence


# =========================================================
# RUN ON FOLDER
# =========================================================
if __name__ == "__main__":
    image_paths = list(Path(TEST_IMAGES_DIR).glob("*"))

    if len(image_paths) == 0:
        print("No images found in test_images/")
    else:
        for path in image_paths:
            if path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                continue

            pred, conf = predict_image(path)
            print(f"{path.name:30} -> {pred:10} ({conf:.3f})")