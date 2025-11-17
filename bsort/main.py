import typer
import yaml
from bsort.train import train_model
from bsort.infer import run_inference
from typing import Optional

app = typer.Typer(help="Bsort: Bottle Cap Detection CLI")

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

@app.command()
def train(config: str = typer.Option(..., help="Path to config.yaml")):
    """Train the YOLO model using parameters from config."""
    cfg = load_config(config)
    typer.echo(f"Starting training for {cfg['project_name']}...")
    train_model(cfg)

@app.command()
def infer(
    config: str = typer.Option(..., help="Path to config.yaml"),
    image: str = typer.Option(..., help="Path to image file"),
):
    """Run inference on a single image."""
    cfg = load_config(config)
    run_inference(cfg, image)

if __name__ == "__main__":
    app()
