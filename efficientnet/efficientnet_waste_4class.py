from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import kagglehub
import numpy as np
import torch
from PIL import Image, ImageFile
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


# =========================
# Config
# =========================
SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS_HEAD = 5
NUM_EPOCHS_FINE = 10
LR_HEAD = 1e-3
LR_FINE = 1e-4
WEIGHT_DECAY = 1e-4
IMAGE_SIZE = 224
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
NUM_WORKERS = 0  # keep 0 on Windows if workers freeze
MODEL_SAVE_PATH = "efficientnet_waste_4class.pt"
REPORT_SAVE_PATH = "dataset_report_waste_4class.json"

UNIFIED_CLASSES = ["trash", "recycle", "organic", "metal"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(UNIFIED_CLASSES)}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DATASETS = [
    {
        "slug": "asdasdasasdas/garbage-classification",
        "name": "garbage_classification_6",
        "exact_map": {
            "cardboard": "recycle",
            "glass": "recycle",
            "metal": "metal",
            "paper": "recycle",
            "plastic": "recycle",
            "trash": "trash",
        },
    },
    {
        "slug": "phenomsg/waste-classification",
        "name": "waste_classification_4plus",
        "exact_map": {
            "hazardous": "trash",
            "hazardous waste": "trash",
            "non recyclable": "trash",
            "non recyclable waste": "trash",
            "organic": "organic",
            "organic waste": "organic",
            "recyclable": "recycle",
            "recyclable waste": "recycle",
            "batteries": "trash",
            "battery": "trash",
            "chemical waste": "trash",
            "medical waste": "trash",
            "plastic wrappers": "trash",
            "styrofoam": "trash",
            "food cups": "trash",
            "food cup": "trash",
            "food waste": "organic",
            "green waste": "organic",
            "paper": "recycle",
            "cardboard": "recycle",
            "glass": "recycle",
            "metal": "metal",
            "plastic": "recycle",
        },
    },
    {
        "slug": "adithyachalla/waste-classification",
        "name": "waste_classification_9",
        "exact_map": {
            "cardboard": "recycle",
            "food organics": "organic",
            "food organic": "organic",
            "glass": "recycle",
            "metal": "metal",
            "miscellaneous trash": "trash",
            "misc trash": "trash",
            "paper": "recycle",
            "plastic": "recycle",
            "textile trash": "trash",
            "vegetation": "organic",
        },
    },
    {
        "slug": "mostafaabla/garbage-classification",
        "name": "garbage_classification_12",
        "exact_map": {
            "paper": "recycle",
            "cardboard": "recycle",
            "biological": "organic",
            "metal": "metal",
            "plastic": "recycle",
            "green glass": "recycle",
            "brown glass": "recycle",
            "white glass": "recycle",
            "glass": "recycle",
            "clothes": "trash",
            "shoes": "trash",
            "batteries": "trash",
            "battery": "trash",
            "trash": "trash",
        },
    },
    {
        "slug": "alistairking/recyclable-and-household-waste-classification",
        "name": "recyclable_household_waste",
        "exact_map": {
            # broad/material labels
            "paper and cardboard": "recycle",
            "paper": "recycle",
            "cardboard": "recycle",
            "glass": "recycle",
            "plastic": "recycle",
            "metal": "metal",
            "organic": "organic",
            "organic waste": "organic",
            "food waste": "organic",
            "compost": "organic",
            "household waste": "trash",
            "general waste": "trash",
            "trash": "trash",
            # subcategory/object aliases that appear in public descriptions
            "aluminum cans": "metal",
            "aluminium cans": "metal",
            "metal cans": "metal",
            "cans": "metal",
            "plastic food containers": "recycle",
            "cardboard boxes": "recycle",
            "food scraps": "organic",
            "eggshells": "organic",
            "coffee grounds": "organic",
            "styrofoam containers": "trash",
            "clothing": "trash",
            "clothes": "trash",
            "shoes": "trash",
            "textile": "trash",
        },
    },
    {
        "slug": "techsash/waste-classification-data",
        "name": "waste_classification_data_2",
        "exact_map": {
            "organic": "organic",
            "recyclable": "recycle",
            "o": "organic",
            "r": "recycle",
        },
    },
    {
        "slug": "reyhanwiranugraha/real-image-dataset",
        "name": "real_image_dataset",
        "exact_map": {
            # likely material labels / object labels
            "cardboard": "recycle",
            "paper": "recycle",
            "plastic": "recycle",
            "glass": "recycle",
            "metal": "metal",
            "aluminium": "metal",
            "aluminum": "metal",
            "can": "metal",
            "cans": "metal",
            "carton": "recycle",
            "bottle": "recycle",
            "bottles": "recycle",
            "food organics": "organic",
            "organic": "organic",
            "vegetation": "organic",
            "miscellaneous trash": "trash",
            "misc": "trash",
            "trash": "trash",
            "textile trash": "trash",
            "textile": "trash",
            "styrofoam": "trash",
            "cup": "trash",
            "cups": "trash",
            "tube": "trash",
        },
    },
]

# dataset-agnostic fallbacks used when exact dataset map does not match.
GLOBAL_RAW_TO_UNIFIED = {
    # metal first so it can override broad recyclable concepts when labels are explicit.
    "metal": "metal",
    "metals": "metal",
    "aluminum": "metal",
    "aluminium": "metal",
    "aluminum can": "metal",
    "aluminium can": "metal",
    "aluminum cans": "metal",
    "aluminium cans": "metal",
    "food can": "metal",
    "food cans": "metal",
    "drink can": "metal",
    "drink cans": "metal",
    "soda can": "metal",
    "soda cans": "metal",
    "tin": "metal",
    "steel": "metal",
    "can": "metal",
    "cans": "metal",

    "organic": "organic",
    "organic waste": "organic",
    "food organics": "organic",
    "food organic": "organic",
    "food waste": "organic",
    "green waste": "organic",
    "biological": "organic",
    "bio": "organic",
    "vegetation": "organic",
    "yard waste": "organic",
    "compost": "organic",
    "compostable": "organic",
    "biodegradable": "organic",

    "recyclable": "recycle",
    "recycle": "recycle",
    "paper": "recycle",
    "newspaper": "recycle",
    "office paper": "recycle",
    "cardboard": "recycle",
    "paper and cardboard": "recycle",
    "glass": "recycle",
    "green glass": "recycle",
    "brown glass": "recycle",
    "white glass": "recycle",
    "plastic": "recycle",
    "plastics": "recycle",
    "bottle": "recycle",
    "bottles": "recycle",
    "carton": "recycle",
    "cartons": "recycle",
    "packaging": "recycle",

    "trash": "trash",
    "general waste": "trash",
    "household waste": "trash",
    "mixed waste": "trash",
    "misc": "trash",
    "miscellaneous": "trash",
    "miscellaneous trash": "trash",
    "textile": "trash",
    "textile trash": "trash",
    "clothes": "trash",
    "clothing": "trash",
    "shoes": "trash",
    "battery": "trash",
    "batteries": "trash",
    "hazardous": "trash",
    "hazardous waste": "trash",
    "medical waste": "trash",
    "chemical waste": "trash",
    "non recyclable": "trash",
    "non recyclable waste": "trash",
    "plastic wrappers": "trash",
    "wrapper": "trash",
    "wrappers": "trash",
    "styrofoam": "trash",
    "foam": "trash",
    "cup": "trash",
    "cups": "trash",
    "tube": "trash",
    "tubes": "trash",
    "food cup": "trash",
    "food cups": "trash",
    "other": "trash",
}

SKIP_ALIASES = {
    "unknown",
    "unidentified",
    "background",
}


ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass(frozen=True)
class ImageRecord:
    image_path: str
    label_name: str
    label_idx: int
    dataset_slug: str
    matched_raw_label: str


class WasteImageDataset(Dataset):
    def __init__(self, records: list[ImageRecord], transform=None):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        image = Image.open(record.image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, record.label_idx


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def normalize_name(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[\\/_\-.]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value


def iter_image_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def match_unified_label(
    image_path: Path,
    dataset_root: Path,
    exact_map: dict[str, str],
) -> tuple[str | None, str | None]:
    """
    Walk upward from the file to the dataset root and use the first ancestor directory
    that looks like a label. This is much more robust than assuming a single ImageFolder root.
    """
    try:
        rel_parts = image_path.relative_to(dataset_root).parts[:-1]
    except ValueError:
        rel_parts = image_path.parts[:-1]

    # Nearest parent first.
    for raw_part in reversed(rel_parts):
        part = normalize_name(raw_part)
        if part in SKIP_ALIASES:
            return None, part
        if part in exact_map:
            return exact_map[part], part
        if part in GLOBAL_RAW_TO_UNIFIED:
            return GLOBAL_RAW_TO_UNIFIED[part], part

    return None, None


def collect_records_for_dataset(
    dataset_slug: str,
    dataset_root: Path,
    exact_map: dict[str, str],
) -> tuple[list[ImageRecord], dict]:
    records: list[ImageRecord] = []
    matched_counter = Counter()
    unified_counter = Counter()
    unresolved_samples = Counter()
    skipped_alias_counter = Counter()
    total_images = 0

    for image_path in iter_image_files(dataset_root):
        total_images += 1
        label_name, matched_raw = match_unified_label(image_path, dataset_root, exact_map)

        if label_name is None:
            if matched_raw is not None:
                skipped_alias_counter[matched_raw] += 1
            else:
                # keep the closest 2 parent dirs to help debugging unmatched structures
                parents = image_path.relative_to(dataset_root).parts[:-1]
                if parents:
                    unresolved_samples[normalize_name(parents[-1])] += 1
            continue

        label_idx = CLASS_TO_IDX[label_name]
        matched_counter[matched_raw] += 1
        unified_counter[label_name] += 1
        records.append(
            ImageRecord(
                image_path=str(image_path),
                label_name=label_name,
                label_idx=label_idx,
                dataset_slug=dataset_slug,
                matched_raw_label=matched_raw,
            )
        )

    summary = {
        "dataset_slug": dataset_slug,
        "dataset_root": str(dataset_root),
        "total_image_files_seen": total_images,
        "usable_images": len(records),
        "usable_by_unified_class": dict(unified_counter),
        "matched_raw_labels": dict(matched_counter),
        "skipped_known_aliases": dict(skipped_alias_counter),
        "top_unresolved_parent_labels": dict(unresolved_samples.most_common(25)),
    }
    return records, summary


def stratified_split(
    records: list[ImageRecord],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[ImageRecord], list[ImageRecord], list[ImageRecord]]:
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("Train/val/test ratios must sum to 1.0")

    rng = random.Random(seed)
    by_class: dict[int, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_class[record.label_idx].append(record)

    train_records: list[ImageRecord] = []
    val_records: list[ImageRecord] = []
    test_records: list[ImageRecord] = []

    for label_idx, group in by_class.items():
        rng.shuffle(group)
        n = len(group)
        if n < 3:
            raise ValueError(
                f"Class '{UNIFIED_CLASSES[label_idx]}' has only {n} image(s); "
                "need at least 3 to create train/val/test splits."
            )

        n_train = max(1, int(round(train_ratio * n)))
        n_val = max(1, int(round(val_ratio * n)))
        n_test = n - n_train - n_val

        # guarantee at least one example in each split
        if n_test <= 0:
            n_test = 1
            if n_train >= n_val and n_train > 1:
                n_train -= 1
            else:
                n_val -= 1

        while n_train + n_val + n_test > n:
            if n_train >= n_val and n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                n_test -= 1

        while n_train + n_val + n_test < n:
            n_train += 1

        train_records.extend(group[:n_train])
        val_records.extend(group[n_train:n_train + n_val])
        test_records.extend(group[n_train + n_val:])

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    rng.shuffle(test_records)
    return train_records, val_records, test_records


def compute_class_weights(records: list[ImageRecord], device: torch.device) -> torch.Tensor:
    counts = torch.tensor(
        [sum(1 for r in records if r.label_idx == idx) for idx in range(len(UNIFIED_CLASSES))],
        dtype=torch.float32,
    )
    if torch.any(counts == 0):
        raise ValueError(f"At least one class has zero training samples: {counts.tolist()}")
    weights = counts.sum() / (len(counts) * counts)
    return weights.to(device)


def counts_by_class(records: list[ImageRecord]) -> dict[str, int]:
    counter = Counter(r.label_name for r in records)
    return {name: counter.get(name, 0) for name in UNIFIED_CLASSES}


def counts_by_dataset(records: list[ImageRecord]) -> dict[str, int]:
    counter = Counter(r.dataset_slug for r in records)
    return dict(counter)


def evaluate(model, loader, criterion, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


def train_one_epoch(model, loader, criterion, optimizer, device: torch.device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
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
    generator = torch.Generator().manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    weights = EfficientNet_B0_Weights.DEFAULT

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_records: list[ImageRecord] = []
    dataset_summaries = []

    print("\n=== Downloading + scanning datasets ===")
    for cfg in DATASETS:
        slug = cfg["slug"]
        print(f"\n[{slug}] downloading...")
        dataset_root = Path(kagglehub.dataset_download(slug))
        print(f"[{slug}] local root: {dataset_root}")

        records, summary = collect_records_for_dataset(
            dataset_slug=slug,
            dataset_root=dataset_root,
            exact_map={normalize_name(k): v for k, v in cfg["exact_map"].items()},
        )
        all_records.extend(records)
        dataset_summaries.append(summary)

        print(f"[{slug}] usable images: {summary['usable_images']} / {summary['total_image_files_seen']}")
        print(f"[{slug}] unified class counts: {summary['usable_by_unified_class']}")
        if summary["top_unresolved_parent_labels"]:
            print(f"[{slug}] top unresolved labels: {summary['top_unresolved_parent_labels']}")

    if not all_records:
        raise RuntimeError("No usable images were found after scanning all datasets.")

    print("\n=== Global dataset summary ===")
    print(f"Total usable images: {len(all_records)}")
    print("Counts by unified class:", counts_by_class(all_records))
    print("Counts by dataset:", counts_by_dataset(all_records))

    # sanity check: make sure every target class exists.
    global_class_counts = counts_by_class(all_records)
    missing_classes = [name for name, count in global_class_counts.items() if count == 0]
    if missing_classes:
        raise RuntimeError(
            f"The merged corpus is missing these target classes: {missing_classes}. "
            "Adjust the label map before training."
        )

    train_records, val_records, test_records = stratified_split(
        all_records,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED,
    )

    print("\n=== Split summary ===")
    print("Train:", len(train_records), counts_by_class(train_records))
    print("Val:  ", len(val_records), counts_by_class(val_records))
    print("Test: ", len(test_records), counts_by_class(test_records))

    train_dataset = WasteImageDataset(train_records, transform=train_transform)
    val_dataset = WasteImageDataset(val_records, transform=eval_transform)
    test_dataset = WasteImageDataset(test_records, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )

    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(UNIFIED_CLASSES))
    model = model.to(device)

    class_weights = compute_class_weights(train_records, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    # Phase 1: train head only.
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=LR_HEAD,
        weight_decay=WEIGHT_DECAY,
    )

    print("\n=== Phase 1: classifier head ===")
    for epoch in range(NUM_EPOCHS_HEAD):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"[Head] Epoch {epoch + 1}/{NUM_EPOCHS_HEAD} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    # Phase 2: fine-tune everything.
    for param in model.features.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR_FINE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_acc = -1.0
    best_state_dict = None

    print("\n=== Phase 2: full fine-tuning ===")
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
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTEST | loss={test_loss:.4f} acc={test_acc:.4f}")

    artifact = {
        "model_state_dict": model.state_dict(),
        "class_names": UNIFIED_CLASSES,
        "class_to_idx": CLASS_TO_IDX,
        "image_size": IMAGE_SIZE,
        "train_counts": counts_by_class(train_records),
        "val_counts": counts_by_class(val_records),
        "test_counts": counts_by_class(test_records),
        "datasets": [cfg["slug"] for cfg in DATASETS],
        "best_val_acc": best_val_acc,
    }
    torch.save(artifact, MODEL_SAVE_PATH)
    print(f"Model saved to: {MODEL_SAVE_PATH}")

    report = {
        "total_usable_images": len(all_records),
        "global_counts": counts_by_class(all_records),
        "split_counts": {
            "train": counts_by_class(train_records),
            "val": counts_by_class(val_records),
            "test": counts_by_class(test_records),
        },
        "counts_by_dataset": counts_by_dataset(all_records),
        "dataset_summaries": dataset_summaries,
    }
    Path(REPORT_SAVE_PATH).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report saved to: {REPORT_SAVE_PATH}")


if __name__ == "__main__":
    main()
