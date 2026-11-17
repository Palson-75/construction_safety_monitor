
from pathlib import Path
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = ROOT / "videos" / "video2.mp4"
PERSON_MODEL_PATH = "yolo26n.pt"
PPE_MODEL_PATH = ROOT / "outputs" / "full_training" / "weights" / "best.pt"

OUTPUT_PATH = ROOT / "outputs" / "video2_combined_test.jpg"

# Load both models
person_model = YOLO(PERSON_MODEL_PATH)
ppe_model = YOLO(str(PPE_MODEL_PATH))

# Read a frame from video2
cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Could not open video2.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * 2))

success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Could not read video frame")

# Run person detection
person_results = person_model.predict(
    frame,
    classes=[0],
    conf=0.15,
    imgsz=1280,
    device=0,
    verbose=False
)[0]

# Run custom PPE detection
ppe_results = ppe_model.predict(
    frame,
    conf=0.15,
    imgsz=640,
    device=0,
    verbose=False
)[0]

# Draw person detections in green
output = frame.copy()

for box in person_results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

    cv2.rectangle(
        output, (x1, y1), (x2, y2),
        (0, 255, 0), 2
    )

    cv2.putText(
        output, "PERSON",
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (0, 255, 0), 2
    )

# Draw custom PPE detections in cyan
for box in ppe_results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    class_name = ppe_model.names[class_id]

    cv2.rectangle(
        output, (x1, y1), (x2, y2),
        (255, 255, 0), 2
    )

    cv2.putText(
        output,
        f"{class_name} {confidence:.2f}",
        (x1, min(y2 + 20, output.shape[0] - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (255, 255, 0), 2
    )

# Print results
print("Person detections:", len(person_results.boxes))
print("PPE detections:", len(ppe_results.boxes))

print("\nPPE classes:")
for box in ppe_results.boxes:
    class_id = int(box.cls[0])
    print(ppe_model.names[class_id])

cv2.imwrite(str(OUTPUT_PATH), output)

print("\nCombined output saved:", OUTPUT_PATH)
