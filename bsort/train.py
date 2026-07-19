import wandb
from ultralytics import YOLO
from typing import Dict, Any

def train_model(cfg: Dict[str, Any]) -> None:
    # Initialize WandB
    wandb.init(project=cfg['project_name'], config=cfg)

    # Load Model
    model = YOLO(cfg['model']['base'])

    # Train
    results = model.train(
        data=cfg['dataset_path'],
        epochs=cfg['training']['epochs'],
        imgsz=cfg['model']['input_size'],
        batch=cfg['training']['batch_size'],
        lr0=cfg['training']['learning_rate'],
        project=cfg['training']['output_dir']
    )

    # Validate
    metrics = model.val()
    wandb.log({"map50": metrics.box.map50, "map50-95": metrics.box.map})

    # Export to ONNX for RPi optimization
    success = model.export(format="onnx", dynamic=False)
    print(f"Model exported to ONNX: {success}")
    
    wandb.finish()
