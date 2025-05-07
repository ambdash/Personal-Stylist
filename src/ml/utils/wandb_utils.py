import wandb
import os
from typing import Optional

def init_wandb(
    project_name: str,
    run_name: Optional[str] = None,
    config: Optional[dict] = None
) -> None:
    """
    Initialize wandb with proper configuration.
    
    Args:
        project_name: Name of the wandb project
        run_name: Optional name for this specific run
        config: Optional configuration dictionary to log
    """
    if wandb.run is not None:
        wandb.finish()
    
    # Initialize wandb
    wandb.init(
        project=project_name,
        name=run_name,
        config=config or {}
    )
    
    # Log system info
    wandb.config.update({
        "python_version": os.popen("python --version").read().strip(),
        "cuda_available": wandb.run.config.get("cuda_available", False),
        "gpu_count": wandb.run.config.get("gpu_count", 0)
    })

def log_metrics(metrics: dict, step: Optional[int] = None) -> None:
    """
    Log metrics to wandb.
    
    Args:
        metrics: Dictionary of metrics to log
        step: Optional step number
    """
    if wandb.run is None:
        raise RuntimeError("wandb not initialized. Call init_wandb first.")
    wandb.log(metrics, step=step) 