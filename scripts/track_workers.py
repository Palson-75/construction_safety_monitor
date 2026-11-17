
from pathlib import Path
from ultralytics import YOLO
import argparse


def main():

    ROOT = Path(__file__).resolve().parents[1]

    MODEL_PATH = (
        ROOT / "outputs" / "full_training" /
        "weights" / "best.pt"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        required=True,
        help="Path to input video"
    )

    args = parser.parse_args()

    video_path = Path(args.source)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    model = YOLO(str(MODEL_PATH))

    results = model.track(
        stream=True,
        source=str(video_path),
        tracker="bytetrack.yaml",
        persist=True,
        conf=0.25,
        iou=0.5,
        imgsz=640,
        device=0,
        show=False,
        save=True,
        project=str(ROOT / "outputs"),
        name="worker_tracking",
        exist_ok=True
    )

    for result in results:
     pass

    print("Tracking completed successfully.")

    print("Tracking completed successfully.")


if __name__ == "__main__":
    main()
