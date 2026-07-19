# Bsort — Bottle Cap Detection & Sorting Pipeline

A machine learning pipeline for detecting and color-sorting bottle caps,
built around YOLOv8, with a focus on making the model deployable on
resource-constrained edge devices (e.g. Raspberry Pi).

## What it does

- **Detection**: fine-tunes a YOLOv8 model to detect bottle caps in images
- **Preprocessing**: classifies each detected cap by color using HSV
  thresholding (`bsort/preprocessing.py`)
- **Training**: configurable training loop with Weights & Biases experiment
  tracking (`bsort/train.py`)
- **Inference**: runs and benchmarks inference on a single image, with a
  latency warning if it exceeds the edge-device budget (`bsort/infer.py`)
- **Export**: exports the trained model to ONNX for faster inference on
  edge hardware

## Project Structure

```
bsort/
├── __init__.py
├── main.py             # CLI entrypoint (typer)
├── train.py             # Training logic
├── infer.py              # Inference logic
├── utils.py               # Color-class label mapping
└── preprocessing.py        # HSV-based color classification
tests/
├── __init__.py
└── test_bsort.py            # Unit tests for color classification
configs/
└── settings.yaml              # Model, training, and inference config
.github/workflows/ci_cd.yml     # Lint, test, and Docker build pipeline
Dockerfile
pyproject.toml
```

## Setup

```bash
pip install .
```

## Usage

Train:
```bash
bsort train --config configs/settings.yaml
```

Run inference on an image:
```bash
bsort infer --config configs/settings.yaml --image path/to/image.jpg
```

## Testing

```bash
pip install .[test]
pytest tests/
```

## CI/CD

GitHub Actions runs formatting checks (black, isort), linting (pylint),
unit tests (pytest), and a Docker build on every push and pull request.

## Notes on edge deployment

The inference step benchmarks latency and warns when it exceeds a 10ms
budget, since the target deployment is a Raspberry Pi. For production use
at that latency target, exporting further to TensorRT or a Hailo-8L
accelerator is recommended over plain ONNX.
