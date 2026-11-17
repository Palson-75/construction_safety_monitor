
from pathlib import Path
from ultralytics import YOLO


def main():
    ROOT = Path(__file__).resolve().parents[1]

    # Fine-tuned model
    MODEL = ROOT / "outputs" / "weak_class_finetune" / "weights" / "best.pt"

    # Keep the original, unchanged test dataset
    DATA = ROOT / "datasets" / "processed_ppe" / "data.yaml"

    if not MODEL.exists():
        raise FileNotFoundError(MODEL)

    model = YOLO(str(MODEL))

    results = model.val(
        data=str(DATA),
        split="test",
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        plots=True,
        project=str(ROOT / "outputs"),
        name="finetune_test_evaluation",
        exist_ok=True
    )

    print("\nFINE-TUNED MODEL TEST EVALUATION")
    print("mAP50:", results.box.map50)
    print("mAP50-95:", results.box.map)

    print("\nCLASS-WISE TEST RESULTS")
    print("-" * 45)

    for i, name in results.names.items():
        print(f"{name:15s}: mAP50-95 = {results.box.maps[i]:.4f}")


if __name__ == "__main__":
    main()
