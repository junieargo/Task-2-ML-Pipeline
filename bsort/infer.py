import time
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, Any

def run_inference(cfg: Dict[str, Any], image_path: str) -> None:
    
    # Load model (Preferably the ONNX exported version for speed)
    # For dev, we use .pt, for prod use .onnx
    model_path = f"{cfg['training']['output_dir']}/weights/best.pt" 
    model = YOLO(model_path) 

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")

    # Warmup
    _ = model(img, verbose=False)

    # Benchmark
    start_time = time.perf_counter()
    results = model(img, conf=cfg['inference']['confidence_threshold'])
    end_time = time.perf_counter()

    inference_ms = (end_time - start_time) * 1000
    print(f"Inference Time: {inference_ms:.2f} ms")

    # Visualize
    res_plotted = results[0].plot()
    cv2.imwrite("result.jpg", res_plotted)
    print("Result saved to result.jpg")

    # Note regarding Edge Device Constraint:
    if inference_ms > 10:
        print("WARNING: Inference exceeded 10ms. Recommend exporting to TensorRT or Hailo-8L for RPi.")
