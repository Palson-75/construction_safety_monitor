
from pathlib import Path
from collections import defaultdict
import cv2
import math
import csv

from ultralytics import YOLO


# =====================================================
# 1. PROJECT SETTINGS
# =====================================================

ROOT = Path(__file__).resolve().parents[1]

VIDEO_DIR = ROOT / "videos"

OUTPUT_DIR = ROOT / "outputs" / "full_video_ppe"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_FILES = [
    f"video{i}.mp4" for i in range(3, 11)
]

PERSON_MODEL_PATH = "yolo26n.pt"

PPE_MODEL_PATH = (
    ROOT / "outputs" / "full_training" /
    "weights" / "best.pt"
)

PERSON_CONF = 0.15
PPE_CONF = 0.15

PERSON_IMGSZ = 1280
PPE_IMGSZ = 640

MIN_EVIDENCE = 5
STATUS_RATIO = 0.80


# =====================================================
# 2. PPE CLASS MAPPING
# =====================================================

CLASS_NAMES = {
    1: ("helmet", "helmet", "positive"),
    2: ("no_helmet", "helmet", "negative"),

    3: ("vest", "vest", "positive"),
    4: ("no_vest", "vest", "negative"),

    5: ("boots", "boots", "positive"),
    6: ("no_boots", "boots", "negative"),

    7: ("gloves", "gloves", "positive"),
    8: ("no_gloves", "gloves", "negative"),

    9: ("goggles", "goggles", "positive"),
    10: ("no_goggle", "goggles", "negative")
}

CATEGORIES = [
    "helmet",
    "vest",
    "boots",
    "gloves",
    "goggles"
]

REGIONS = {
    "helmet": (0.00, 0.35),
    "vest": (0.20, 0.72),
    "boots": (0.65, 1.00),
    "gloves": (0.15, 0.85),
    "goggles": (0.00, 0.40)
}


# =====================================================
# 3. ASSOCIATION FUNCTION
# =====================================================

def association_score(worker_box, point, category):

    x1, y1, x2, y2 = worker_box

    width = max(x2 - x1, 1)
    height = max(y2 - y1, 1)

    px, py = point

    relative_x = (px - x1) / width
    relative_y = (py - y1) / height

    top, bottom = REGIONS[category]

    if not (0.05 <= relative_x <= 0.95):
        return None

    if not (top <= relative_y <= bottom):
        return None

    region_center = (top + bottom) / 2

    return math.sqrt(
        (relative_x - 0.5) ** 2 +
        (relative_y - region_center) ** 2
    )


# =====================================================
# 4. STATUS FUNCTION
# =====================================================

def get_status(counts):

    positive = counts["positive"]
    negative = counts["negative"]

    total = positive + negative

    if total < MIN_EVIDENCE:
        return "UNKNOWN"

    positive_ratio = positive / total
    negative_ratio = negative / total

    if positive_ratio >= STATUS_RATIO:
        return "DETECTED"

    if negative_ratio >= STATUS_RATIO:
        return "NEGATIVE"

    return "CONFLICT"


# =====================================================
# 5. PROCESS ONE VIDEO
# =====================================================

def process_video(video_name, ppe_model):

    video_path = VIDEO_DIR / video_name

    if not video_path.exists():
        print(f"\nSKIPPING: {video_name} not found")
        return False

    video_label = Path(video_name).stem

    output_video = (
        OUTPUT_DIR / f"{video_label}_ppe_monitor.avi"
    )

    output_csv = (
        OUTPUT_DIR / f"worker_ppe_report_{video_label}.csv"
    )

    print("\n" + "=" * 55)
    print(f"PROCESSING {video_name}")
    print("=" * 55)

    # Create a fresh person model for each video.
    # This keeps ByteTrack IDs separate between videos.
    person_model = YOLO(PERSON_MODEL_PATH)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Cannot open {video_name}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 25

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        print("Could not create output video")
        return False

    worker_evidence = defaultdict(
        lambda: defaultdict(
            lambda: {
                "positive": 0,
                "negative": 0
            }
        )
    )

    worker_frames_seen = defaultdict(int)

    frame_count = 0

    try:

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_count += 1
            output = frame.copy()

            # -----------------------------------------
            # PERSON DETECTION AND TRACKING
            # -----------------------------------------

            person_result = person_model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                conf=PERSON_CONF,
                imgsz=PERSON_IMGSZ,
                device=0,
                verbose=False
            )[0]

            workers = []

            if (
                person_result.boxes is not None
                and person_result.boxes.id is not None
            ):

                boxes = person_result.boxes.xyxy.cpu().numpy()

                ids = (
                    person_result.boxes.id.cpu()
                    .numpy().astype(int)
                )

                for box, track_id in zip(boxes, ids):

                    x1, y1, x2, y2 = map(int, box)

                    track_id = int(track_id)

                    workers.append({
                        "id": track_id,
                        "box": (x1, y1, x2, y2)
                    })

                    worker_evidence[track_id]
                    worker_frames_seen[track_id] += 1

            # -----------------------------------------
            # PPE DETECTION
            # -----------------------------------------

            ppe_result = ppe_model.predict(
                frame,
                conf=PPE_CONF,
                imgsz=PPE_IMGSZ,
                device=0,
                verbose=False
            )[0]

            # One vote per worker/category/frame
            frame_votes = {}

            if ppe_result.boxes is not None:

                for box in ppe_result.boxes:

                    class_id = int(box.cls[0])

                    if class_id not in CLASS_NAMES:
                        continue

                    class_name, category, polarity = (
                        CLASS_NAMES[class_id]
                    )

                    px1, py1, px2, py2 = map(
                        int, box.xyxy[0].tolist()
                    )

                    center = (
                        (px1 + px2) / 2,
                        (py1 + py2) / 2
                    )

                    confidence = float(box.conf[0])

                    candidates = []

                    for worker in workers:

                        score = association_score(
                            worker["box"],
                            center,
                            category
                        )

                        if score is not None:
                            candidates.append(
                                (score, worker)
                            )

                    candidates.sort(
                        key=lambda item: item[0]
                    )

                    if not candidates:
                        continue

                    if (
                        len(candidates) > 1
                        and candidates[1][0] -
                        candidates[0][0] < 0.08
                    ):
                        continue

                    track_id = candidates[0][1]["id"]

                    key = (track_id, category)

                    if (
                        key not in frame_votes
                        or confidence >
                        frame_votes[key]["confidence"]
                    ):

                        frame_votes[key] = {
                            "confidence": confidence,
                            "polarity": polarity
                        }

            # -----------------------------------------
            # UPDATE WORKER EVIDENCE
            # -----------------------------------------

            for (track_id, category), vote in frame_votes.items():

                polarity = vote["polarity"]

                worker_evidence[track_id][category][polarity] += 1

            # -----------------------------------------
            # DRAW WORKER INFORMATION
            # -----------------------------------------

            for worker in workers:

                track_id = worker["id"]

                x1, y1, x2, y2 = worker["box"]

                evidence = worker_evidence[track_id]

                statuses = {
                    category: get_status(evidence[category])
                    for category in CATEGORIES
                }

                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    output,
                    f"Worker ID: {track_id}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

                details = (
                    f"H:{statuses['helmet']} "
                    f"V:{statuses['vest']} "
                    f"B:{statuses['boots']}"
                )

                cv2.putText(
                    output,
                    details,
                    (x1, min(y2 + 18, height - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 0),
                    1
                )

            writer.write(output)

            if frame_count % 50 == 0:
                print(f"{video_name}: {frame_count} frames")

    finally:

        cap.release()
        writer.release()

    # ---------------------------------------------
    # GENERATE CSV REPORT
    # ---------------------------------------------

    fieldnames = [
        "worker_id",
        "frames_seen",
        "ppe_category",
        "positive_detections",
        "negative_detections",
        "total_detections",
        "positive_ratio",
        "final_status"
    ]

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        csv_writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        csv_writer.writeheader()

        for track_id in sorted(worker_frames_seen):

            for category in CATEGORIES:

                counts = worker_evidence[track_id][category]

                positive = counts["positive"]
                negative = counts["negative"]

                total = positive + negative

                ratio = (
                    positive / total if total > 0 else 0
                )

                csv_writer.writerow({

                    "worker_id": track_id,

                    "frames_seen": worker_frames_seen[track_id],

                    "ppe_category": category,

                    "positive_detections": positive,

                    "negative_detections": negative,

                    "total_detections": total,

                    "positive_ratio": round(ratio, 3),

                    "final_status": get_status(counts)
                })

    # ---------------------------------------------
    # FINAL VIDEO SUMMARY
    # ---------------------------------------------

    print(f"\nCompleted: {video_name}")
    print("Frames processed:", frame_count)
    print("Workers tracked:", len(worker_frames_seen))
    print("Video saved:", output_video)
    print("CSV saved:", output_csv)

    return True


# =====================================================
# 6. MAIN PROGRAM
# =====================================================

def main():

    if not PPE_MODEL_PATH.exists():
        raise FileNotFoundError(PPE_MODEL_PATH)

    print("Loading PPE model...")

    ppe_model = YOLO(str(PPE_MODEL_PATH))

    print("PPE model loaded.")

    successful = []
    skipped = []

    for video_name in VIDEO_FILES:

        try:

            result = process_video(
                video_name,
                ppe_model
            )

            if result:
                successful.append(video_name)
            else:
                skipped.append(video_name)

        except Exception as error:

            print(f"\nERROR processing {video_name}: {error}")
            skipped.append(video_name)

    print("\n" + "=" * 55)
    print("BATCH PROCESSING COMPLETED")
    print("=" * 55)

    print("Successfully processed:", successful)
    print("Skipped or failed:", skipped)

    print("\nOutput folder:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
