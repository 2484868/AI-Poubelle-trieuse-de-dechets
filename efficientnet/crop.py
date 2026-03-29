import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


# =========================================================
# CONFIG
# =========================================================
INPUT_DIR = "efficientnet/raw_images"
OUTPUT_DIR = "efficientnet/cropped_images"

MODEL_ID = "IDEA-Research/grounding-dino-base"

# Prompts must be lowercase and end with a dot.
# Try changing this if detection is weak.
TEXT_PROMPT = "centered object. piece of trash. waste item. object on plate. garbage. recyclable item. compost item. bottle. can. paper. food waste. trash."

BOX_THRESHOLD = 0.28
TEXT_THRESHOLD = 0.20

PADDING_RATIO = 0.08  # extra crop margin around the detected box

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =========================================================
# HELPERS
# =========================================================
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)
    model.eval()
    return processor, model, device


def clamp(value, low, high):
    return max(low, min(value, high))


def add_padding_to_box(box, image_width, image_height, padding_ratio=0.08):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)

    x1 = clamp(int(x1) - pad_x, 0, image_width)
    y1 = clamp(int(y1) - pad_y, 0, image_height)
    x2 = clamp(int(x2) + pad_x, 0, image_width)
    y2 = clamp(int(y2) + pad_y, 0, image_height)

    return [x1, y1, x2, y2]


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def choose_best_detection(boxes, scores, labels):
    """
    Current strategy:
    - highest confidence first
    - if scores are tied or close, larger box usually wins indirectly less often
    """
    if len(boxes) == 0:
        return None

    best_idx = max(range(len(boxes)), key=lambda i: float(scores[i]))
    return boxes[best_idx], scores[best_idx], labels[best_idx]


def detect_best_box(image, processor, model, device):
    inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        target_sizes=[image.size[::-1]],
    )[0]

    boxes = results["boxes"]
    scores = results["scores"]
    labels = results["labels"]

    # 🔴 MANUAL FILTER (this replaces box_threshold)
    filtered = [
        (box, score, label)
        for box, score, label in zip(boxes, scores, labels)
        if float(score) > BOX_THRESHOLD
    ]

    if len(filtered) == 0:
        return None

    # pick best
    box, score, label = max(filtered, key=lambda x: float(x[1]))

    box = [float(v) for v in box.tolist()]
    return box, float(score), str(label)

def crop_and_save_image(image_path, output_dir, processor, model, device):
    image = Image.open(image_path).convert("RGB")
    detection = detect_best_box(image, processor, model, device)

    if detection is None:
        print(f"[NO DETECTION] {image_path.name}")
        return False

    box, score, label = detection
    x1, y1, x2, y2 = add_padding_to_box(
        box,
        image_width=image.width,
        image_height=image.height,
        padding_ratio=PADDING_RATIO,
    )

    cropped = image.crop((x1, y1, x2, y2))

    output_path = output_dir / image_path.name
    cropped.save(output_path)

    print(
        f"[OK] {image_path.name} -> {output_path.name} | "
        f"label='{label}' score={score:.3f} box=({x1},{y1},{x2},{y2})"
    )
    return True


def main():
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    processor, model, device = load_model()
    print(f"Using device: {device}")
    print(f"Prompt: {TEXT_PROMPT}")

    image_paths = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]

    if not image_paths:
        print("No images found.")
        return

    success = 0
    failed = 0

    for image_path in image_paths:
        try:
            ok = crop_and_save_image(image_path, output_dir, processor, model, device)
            if ok:
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[ERROR] {image_path.name}: {e}")
            failed += 1

    print(f"\nDone. Cropped: {success}, Failed: {failed}")


if __name__ == "__main__":
    main()