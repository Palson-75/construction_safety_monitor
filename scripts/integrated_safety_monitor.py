
from pathlib import Path
from collections import defaultdict
import math
import cv2
import joblib
import pandas as pd
from ultralytics import YOLO

# =====================================================
# 1. PROJECT PATHS AND SETTINGS
# =====================================================

ROOT = Path(__file__).resolve().parents[1]

VIDEO_DIR = ROOT / "videos"
OUTPUT_DIR = ROOT / "outputs" / "integrated_monitor"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PERSON_MODEL_PATH = "yolo26n.pt"

PPE_MODEL_PATH = (
    ROOT / "outputs" / "full_training" /
    "weights" / "best.pt"
)

XGB_MODEL_PATH = ROOT / "models" / "xgboost_model.pkl"
FEATURES_PATH = ROOT / "models" / "xgboost_features.pkl"

VIDEO_NAME = "video2.mp4"

PERSON_CONF = 0.15
PPE_CONF = 0.15

PERSON_IMGSZ = 1280
PPE_IMGSZ = 640

MIN_EVIDENCE = 5
STATUS_RATIO = 0.80

CATEGORIES = [
    "helmet", "vest", "boots", "gloves", "goggles"
]


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

REGIONS = {
    "helmet": (0.00, 0.35),
    "vest": (0.20, 0.72),
    "boots": (0.65, 1.00),
    "gloves": (0.15, 0.85),
    "goggles": (0.00, 0.40)
}


# =====================================================
# 3. DRAW LABELS AND BOUNDING BOXES
# =====================================================

def draw_label(frame, box, label, color, thickness=2):

    height, width = frame.shape[:2]

    x1, y1, x2, y2 = map(int, box)

    # Keep coordinates inside the frame
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))

    # Draw bounding box
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        thickness
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.50

    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness
    )

    # Put label above bounding box when possible
    if y1 - text_height - 10 >= 0:
        label_top = y1 - text_height - 10
        text_y = y1 - 6
    else:
        label_top = y1
        text_y = min(
            y1 + text_height + 8,
            height - baseline - 2
        )

    label_left = x1

    label_right = min(
        x1 + text_width + 10,
        width - 1
    )

    label_bottom = min(
        label_top + text_height + baseline + 8,
        height - 1
    )

    # Draw filled label background
    cv2.rectangle(
        frame,
        (label_left, label_top),
        (label_right, label_bottom),
        color,
        -1
    )

    # Draw label text
    cv2.putText(
        frame,
        label,
        (label_left + 4, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


# =====================================================
# 4. PPE-TO-WORKER ASSOCIATION
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
# 5. PPE STATUS
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
# 6. LOAD MODELS
# =====================================================

def load_models():

    for path in [
        PPE_MODEL_PATH,
        XGB_MODEL_PATH,
        FEATURES_PATH
    ]:

        if not Path(path).exists():
            raise FileNotFoundError(
                f"Missing file: {path}"
            )

    print("Loading person detection model...")
    person_model = YOLO(PERSON_MODEL_PATH)

    print("Loading PPE detection model...")
    ppe_model = YOLO(str(PPE_MODEL_PATH))

    print("Loading XGBoost model...")
    xgb_model = joblib.load(XGB_MODEL_PATH)

    feature_columns = joblib.load(FEATURES_PATH)

    print("Models loaded successfully.")
    print("Expected XGBoost features:", len(feature_columns))

    if len(feature_columns) != 21:
        raise ValueError(
            f"Expected 21 features, found {len(feature_columns)}"
        )

    return (
        person_model,
        ppe_model,
        xgb_model,
        feature_columns
    )


# =====================================================
# 7. BUILD EXACT XGBOOST FEATURES
# =====================================================

def build_worker_features(frames_seen, evidence):

    row = {
        "frames_seen": frames_seen
    }

    statuses = {}

    for category in CATEGORIES:

        counts = evidence[category]

        positive = counts["positive"]
        negative = counts["negative"]

        total = positive + negative

        ratio = (
            positive / total
            if total > 0
            else 0
        )

        row[f"{category}_positive"] = positive
        row[f"{category}_negative"] = negative
        row[f"{category}_total"] = total
        row[f"{category}_positive_ratio"] = ratio

        statuses[category] = get_status(counts)

    return row, statuses


# =====================================================
# 8. RULE-BASED EVIDENCE GATE + XGBOOST
# =====================================================

def predict_risk(
    feature_row,
    statuses,
    xgb_model,
    feature_columns
):

    helmet_status = statuses["helmet"]
    vest_status = statuses["vest"]

    # Match the rules used to create the training labels.
    if (
        helmet_status == "NEGATIVE"
        or vest_status == "NEGATIVE"
    ):
        return "High"

    if (
        helmet_status != "DETECTED"
        or vest_status != "DETECTED"
    ):
        return "Insufficient Evidence"

    # Only predict when the required evidence exists.
    input_df = pd.DataFrame(
        [[feature_row[column] for column in feature_columns]],
        columns=feature_columns
    )

    prediction = int(
        xgb_model.predict(input_df)[0]
    )

    return "Low" if prediction == 0 else "High"


# =====================================================
# 9. PROCESS VIDEO
# =====================================================

def process_video(
    video_name,
    person_model,
    ppe_model,
    xgb_model,
    feature_columns
):

    video_path = VIDEO_DIR / video_name

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    output_video = (
        OUTPUT_DIR /
        f"{Path(video_name).stem}_integrated.avi"
    )

    output_csv = (
        OUTPUT_DIR /
        f"{Path(video_name).stem}_risk_report.csv"
    )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

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
        raise RuntimeError(
            "Could not create output video."
        )

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
            # PERSON DETECTION + TRACKING
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

                boxes = (
                    person_result.boxes.xyxy.cpu().numpy()
                )

                ids = (
                    person_result.boxes.id.cpu()
                    .numpy()
                    .astype(int)
                )

                confidences = (
                    person_result.boxes.conf.cpu().numpy()
                )

                for box, track_id, confidence in zip(
                    boxes,
                    ids,
                    confidences
                ):

                    x1, y1, x2, y2 = map(
                        int,
                        box
                    )

                    track_id = int(track_id)

                    workers.append({
                        "id": track_id,
                        "box": (x1, y1, x2, y2),
                        "confidence": float(confidence)
                    })

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
                        int,
                        box.xyxy[0].tolist()
                    )

                    confidence = float(box.conf[0])

                    # ---------------------------------
                    # DRAW PPE DETECTION LABELS
                    # ---------------------------------

                    if polarity == "positive":
                        ppe_color = (0, 200, 0)
                    else:
                        ppe_color = (0, 0, 255)

                    draw_label(
                        output,
                        (px1, py1, px2, py2),
                        f"{class_name} {confidence:.2f}",
                        ppe_color
                    )

                    # ---------------------------------
                    # PPE-TO-WORKER ASSOCIATION
                    # ---------------------------------

                    center = (
                        (px1 + px2) / 2,
                        (py1 + py2) / 2
                    )

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
                        and candidates[1][0] - candidates[0][0] < 0.08
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
            # UPDATE EVIDENCE
            # -----------------------------------------

            for (
                track_id,
                category
            ), vote in frame_votes.items():

                worker_evidence[track_id][category][
                    vote["polarity"]
                ] += 1

            # -----------------------------------------
            # DRAW PERSON AND RISK LABELS
            # -----------------------------------------

            for worker in workers:

                track_id = worker["id"]

                x1, y1, x2, y2 = worker["box"]

                person_confidence = worker["confidence"]

                feature_row, statuses = build_worker_features(
                    worker_frames_seen[track_id],
                    worker_evidence[track_id]
                )

                risk = predict_risk(
                    feature_row,
                    statuses,
                    xgb_model,
                    feature_columns
                )

                if risk == "High":
                    box_color = (0, 0, 255)

                elif risk == "Low":
                    box_color = (0, 200, 0)

                else:
                    box_color = (0, 165, 255)

                # Draw person box and label
                draw_label(
                    output,
                    (x1, y1, x2, y2),
                    (
                        f"Person {person_confidence:.2f} | "
                        f"ID:{track_id} | {risk}"
                    ),
                    box_color
                )

                # Display helmet and vest evidence status
                details = (
                    f"Helmet:{statuses['helmet']} "
                    f"Vest:{statuses['vest']}"
                )

                detail_y = min(
                    y2 + 20,
                    height - 8
                )

                cv2.putText(
                    output,
                    details,
                    (max(x1, 5), detail_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA
                )

            # -----------------------------------------
            # SAVE ANNOTATED FRAME
            # -----------------------------------------

            writer.write(output)

            if frame_count % 50 == 0:
                print(
                    f"Processed frames: {frame_count}"
                )

    finally:

        cap.release()
        writer.release()

    # =================================================
    # 10. SAVE WORKER RISK REPORT
    # =================================================

    report_rows = []

    for track_id in sorted(worker_frames_seen):

        feature_row, statuses = build_worker_features(
            worker_frames_seen[track_id],
            worker_evidence[track_id]
        )

        risk = predict_risk(
            feature_row,
            statuses,
            xgb_model,
            feature_columns
        )

        report_rows.append({
            "worker_id": track_id,
            "frames_seen": worker_frames_seen[track_id],
            **statuses,
            **feature_row,
            "risk_level": risk
        })

    pd.DataFrame(report_rows).to_csv(
        output_csv,
        index=False
    )

    print("\n========== PROCESSING COMPLETED ==========")

    print("Video:", video_name)
    print("Frames processed:", frame_count)
    print("Workers tracked:", len(worker_frames_seen))

    print("Annotated video:", output_video)
    print("Risk report:", output_csv)

    if report_rows:

        print("\nRisk summary:")

        print(
            pd.Series(
                [
                    row["risk_level"]
                    for row in report_rows
                ]
            ).value_counts()
        )


# =====================================================
# 11. MAIN
# =====================================================

def main():

    person_model, ppe_model, xgb_model, feature_columns = (
        load_models()
    )

    process_video(
        VIDEO_NAME,
        person_model,
        ppe_model,
        xgb_model,
        feature_columns
    )


if __name__ == "__main__":
    main()
