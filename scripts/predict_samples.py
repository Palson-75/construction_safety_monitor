
from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "outputs" / "yolo_smoke_test" / "weights" / "best.pt"
IMAGE_DIR = ROOT / "datasets" / "processed_ppe" / "images" / "val"
OUTPUT_DIR = ROOT / "outputs" / "prediction_samples"

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    model = YOLO(str(MODEL_PATH))

    results = model.predict(
        source=str(IMAGE_DIR),
        imgsz=640,
        conf=0.15,
        device=0,
        save=True,
        project=str(ROOT / "outputs"),
        name="prediction_samples",
        exist_ok=True,
        max_det=30
    )

    print("\nPrediction completed!")
    print("Results saved at:", OUTPUT_DIR)

    for result in results[:5]:
        print(
            result.path,
            "Detections:", len(result.boxes)
        )

if __name__ == "__main__":
    main()
