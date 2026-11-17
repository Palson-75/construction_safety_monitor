
from pathlib import Path
import cv2
import math
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = ROOT / "videos" / "video2.mp4"
PERSON_MODEL_PATH = "yolo26n.pt"
PPE_MODEL_PATH = (
    ROOT / "outputs" / "full_training" / "weights" / "best.pt"
)

OUTPUT_PATH = ROOT / "outputs" / "video2_ppe_association_v2.jpg"

# Load models
person_model = YOLO(PERSON_MODEL_PATH)
ppe_model = YOLO(str(PPE_MODEL_PATH))

# Read a frame at 2 seconds
cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Could not open video2.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * 2))

success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Could not read video frame")

# Detect people
person_results = person_model.predict(
    frame,
    classes=[0],
    conf=0.15,
    imgsz=1280,
    device=0,
    verbose=False
)[0]

# Detect PPE
ppe_results = ppe_model.predict(
    frame,
    conf=0.15,
    imgsz=640,
    device=0,
    verbose=False
)[0]

workers = []

for index, box in enumerate(person_results.boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

    workers.append({
        "id": index + 1,
        "box": (x1, y1, x2, y2),
        "status": {
            "helmet": [],
            "vest": [],
            "boots": []
        }
    })

# Class IDs from our custom dataset
class_names = {
    1: "helmet",
    2: "no_helmet",
    3: "vest",
    4: "no_vest",
    5: "boots",
    6: "no_boots",
    7: "gloves",
    8: "no_gloves",
    9: "goggles",
    10: "no_goggle"
}

# PPE class -> compliance category
category_map = {
    "helmet": "helmet",
    "no_helmet": "helmet",
    "vest": "vest",
    "no_vest": "vest",
    "boots": "boots",
    "no_boots": "boots"
}

# Normalized vertical regions within a worker box
regions = {
    "helmet": (0.00, 0.35),
    "vest": (0.20, 0.72),
    "boots": (0.65, 1.00)
}


def get_association_score(worker_box, point, category):
    x1, y1, x2, y2 = worker_box

    width = max(x2 - x1, 1)
    height = max(y2 - y1, 1)

    px, py = point

    # Convert PPE center to worker-relative coordinates
    relative_x = (px - x1) / width
    relative_y = (py - y1) / height

    top, bottom = regions[category]

    # Require PPE center to lie inside the relevant body region
    if not (0.10 <= relative_x <= 0.90):
        return None

    if not (top <= relative_y <= bottom):
        return None

    # Score proximity to the center of the expected body region
    region_center_y = (top + bottom) / 2

    score = math.sqrt(
        (relative_x - 0.5) ** 2 +
        (relative_y - region_center_y) ** 2
    )

    return score


# Associate PPE detections
for box in ppe_results.boxes:

    px1, py1, px2, py2 = map(
        int, box.xyxy[0].tolist()
    )

    class_id = int(box.cls[0])
    class_name = class_names.get(class_id)

    if class_name not in category_map:
        continue

    category = category_map[class_name]

    center = (
        (px1 + px2) / 2,
        (py1 + py2) / 2
    )

    candidates = []

    for worker in workers:

        score = get_association_score(
            worker["box"],
            center,
            category
        )

        if score is not None:
            candidates.append((score, worker))

    if not candidates:
        print(f"{class_name}: UNKNOWN association")
        continue

    candidates.sort(key=lambda item: item[0])

    # If two workers are similarly plausible, avoid guessing
    if (
        len(candidates) > 1
        and candidates[1][0] - candidates[0][0] < 0.08
    ):
        print(f"{class_name}: AMBIGUOUS association")
        continue

    best_score, assigned_worker = candidates[0]

    assigned_worker["status"][category].append(class_name)

    print(
        f"{class_name} assigned to Worker "
        f"{assigned_worker['id']} "
        f"(score: {best_score:.3f})"
    )


# Determine status without assuming undetected PPE is missing
def get_status(detections, positive, negative):

    has_positive = positive in detections
    has_negative = negative in detections

    if has_positive and has_negative:
        return "CONFLICT"

    if has_positive:
        return "DETECTED"

    if has_negative:
        return "NEGATIVE DETECTION"

    return "UNKNOWN"


# Draw results
output = frame.copy()

for worker in workers:

    x1, y1, x2, y2 = worker["box"]

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    statuses = worker["status"]

    helmet_status = get_status(
        statuses["helmet"], "helmet", "no_helmet"
    )

    vest_status = get_status(
        statuses["vest"], "vest", "no_vest"
    )

    boots_status = get_status(
        statuses["boots"], "boots", "no_boots"
    )

    print(f"\nWorker {worker['id']}")
    print("Helmet:", helmet_status)
    print("Vest:", vest_status)
    print("Boots:", boots_status)

    label = (
        f"Worker {worker['id']} | "
        f"H:{helmet_status} "
        f"V:{vest_status} "
        f"B:{boots_status}"
    )

    cv2.putText(
        output,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 255, 0),
        1
    )

cv2.imwrite(str(OUTPUT_PATH), output)

print("\nTotal workers:", len(workers))
print("Output saved:", OUTPUT_PATH)
