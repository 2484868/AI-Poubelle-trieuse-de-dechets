import os
import random
from pathlib import Path

import kagglehub
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS_HEAD = 5
NUM_EPOCHS_FINE = 10
LR_HEAD = 1e-3
LR_FINE = 1e-4
IMAGE_SIZE = 224
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
NUM_WORKERS = 0  # start with 0 on Windows if you get worker issues

KAGGLE_DATASET = "asdasdasasdas/garbage-classification"
DATASET_SUBFOLDER = "Garbage classification/Garbage classification"
MODEL_SAVE_PATH = "efficientnet_trash_classifier.pt"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def contains_class_folders(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False

    subdirs = [p for p in path.iterdir() if p.is_dir()]
    if len(subdirs) < 2:
        return False

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    class_like = 0

    for subdir in subdirs:
        has_image = any(
            file.is_file() and file.suffix.lower() in image_exts
            for file in subdir.iterdir()
        )
        if has_image:
            class_like += 1

    return class_like >= 2


def find_imagefolder_root(downloaded_path: Path, subfolder: str | None = None) -> Path:
    if subfolder is not None:
        candidate = downloaded_path / subfolder
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Subfolder '{subfolder}' not found inside {downloaded_path}")

    if contains_class_folders(downloaded_path):
        return downloaded_path

    for root, dirs, files in os.walk(downloaded_path):
        root_path = Path(root)
        if contains_class_folders(root_path):
            return root_path

    raise FileNotFoundError(
        "Could not find a valid ImageFolder root. "
        "You likely need to set DATASET_SUBFOLDER."
    )


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


def main():
    set_seed(SEED)
    g = torch.Generator()
    g.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    weights = EfficientNet_B0_Weights.DEFAULT

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    print("Downloading dataset from Kaggle...")
    downloaded_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    print(f"Downloaded to: {downloaded_path}")

    data_dir = find_imagefolder_root(downloaded_path, DATASET_SUBFOLDER)
    print(f"Using ImageFolder root: {data_dir}")

    full_dataset = datasets.ImageFolder(root=data_dir)
    class_names = full_dataset.classes
    num_classes = len(class_names)

    print("Classes:", class_names)
    print("Total images:", len(full_dataset))

    n_total = len(full_dataset)
    n_train = int(TRAIN_RATIO * n_total)
    n_val = int(VAL_RATIO * n_total)
    n_test = n_total - n_train - n_val

    train_subset, val_subset, test_subset = random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=g
    )

    train_dataset = TransformedSubset(train_subset, transform=train_transform)
    val_dataset = TransformedSubset(val_subset, transform=eval_transform)
    test_dataset = TransformedSubset(test_subset, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=g
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=g
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=g
    )

    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=LR_HEAD)

    print("\n=== Phase 1 ===")
    for epoch in range(NUM_EPOCHS_HEAD):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"[Head] Epoch {epoch + 1}/{NUM_EPOCHS_HEAD} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    for param in model.features.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_FINE)

    print("\n=== Phase 2 ===")
    best_val_acc = -1.0
    best_state_dict = None

    for epoch in range(NUM_EPOCHS_FINE):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"[Fine] Epoch {epoch + 1}/{NUM_EPOCHS_FINE} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTEST | loss={test_loss:.4f} acc={test_acc:.4f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "image_size": IMAGE_SIZE,
    }, MODEL_SAVE_PATH)

    print(f"Model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()