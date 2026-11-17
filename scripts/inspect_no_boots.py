
from pathlib import Path
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "outputs" / "full_training" / "weights" / "best.pt"
DATASET = ROOT / "datasets" / "processed_ppe"

IMAGE_DIR = DATASET / "images" / "test"
LABEL_DIR = DATASET / "labels" / "test"

OUTPUT_DIR = ROOT / "outputs" / "no_boots_error_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "person", "helmet", "no_helmet", "vest", "no_vest",
    "boots", "no_boots", "gloves", "no_gloves",
    "goggles", "no_goggle"
]

TARGET_CLASS = 6

model = YOLO(str(MODEL_PATH))

# Find test images containing no_boots annotations
image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

selected_images = []

for label_file in LABEL_DIR.glob("*.txt"):
    lines = label_file.read_text().splitlines()

    contains_target = False

    for line in lines:
        parts = line.split()

        if len(parts) == 5:
            try:
                if int(parts[0]) == TARGET_CLASS:
                    contains_target = True
                    break
            except ValueError:
                pass

    if contains_target:
        for ext in image_extensions:
            image_path = IMAGE_DIR / (label_file.stem + ext)

            if image_path.exists():
                selected_images.append((image_path, label_file))
                break

print("Images containing no_boots:", len(selected_images))

for image_path, label_file in selected_images:

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    ground_truth = image.copy()
    predictions = image.copy()

    height, width = image.shape[:2]

    # Draw ground-truth annotations
    for line in label_file.read_text().splitlines():

        parts = line.split()

        if len(parts) != 5:
            continue

        cls = int(parts[0])
        x, y, w, h = map(float, parts[1:])

        x1 = int((x - w / 2) * width)
        y1 = int((y - h / 2) * height)
        x2 = int((x + w / 2) * width)
        y2 = int((y + h / 2) * height)

        cv2.rectangle(
            ground_truth, (x1, y1), (x2, y2), (0, 255, 0), 2
        )

        cv2.putText(
            ground_truth,
            CLASS_NAMES[cls],
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Run original model with a lower confidence threshold
    result = model.predict(
        source=image,
        imgsz=640,
        conf=0.10,
        iou=0.5,
        device=0,
        verbose=False
    )[0]

    # Draw model predictions
    for box in result.boxes:

        cls = int(box.cls.item())
        confidence = float(box.conf.item())

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        cv2.rectangle(
            predictions, (x1, y1), (x2, y2), (0, 165, 255), 2
        )

        label = f"{CLASS_NAMES[cls]} {confidence:.2f}"

        cv2.putText(
            predictions,
            label,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2
        )

    # Put both images side by side
    cv2.putText(
        ground_truth, "GROUND TRUTH",
        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
    )

    cv2.putText(
        predictions, "MODEL PREDICTIONS",
        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2
    )

    comparison = cv2.hconcat([ground_truth, predictions])

    output_path = OUTPUT_DIR / f"{image_path.stem}_comparison.jpg"

    cv2.imwrite(str(output_path), comparison)

    print("Saved:", output_path)

print("\nNO_BOOTS INSPECTION COMPLETED")
