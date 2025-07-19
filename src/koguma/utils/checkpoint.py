"""Checkpoint utilities for saving and loading models."""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from accelerate import Accelerator
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def save_checkpoint(
    model,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: Path,
    accelerator: Optional[Accelerator] = None,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    loss: Optional[float] = None,
    metrics: Optional[Dict[str, Any]] = None
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        output_dir: Directory to save checkpoint
        accelerator: Accelerator instance (optional)
        epoch: Current epoch (optional)
        step: Current step (optional)
        loss: Current loss (optional)
        metrics: Additional metrics to save (optional)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    if accelerator is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.save(unwrapped_model.state_dict(), output_dir / "pytorch_model.bin")
    else:
        torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
    
    # Save model config
    if hasattr(model, 'config'):
        model.config.save_pretrained(output_dir)
    
    # Save tokenizer
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    
    # Save training state
    training_state = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "metrics": metrics or {}
    }
    
    with open(output_dir / "training_state.json", "w") as f:
        json.dump(training_state, f, indent=2)
    
    logger.info(f"Checkpoint saved to {output_dir}")


def load_checkpoint(
    checkpoint_path: Path,
    model_config=None,
    device: str = "cuda",
    strict: bool = True
):
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        model_config: Model configuration (optional)
        device: Device to load model to
        strict: Whether to strictly enforce state dict loading
        
    Returns:
        Loaded model
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Load model config if not provided
    if model_config is None:
        from koguma.models import KogumaConfig
        config_path = checkpoint_path / "config.json"
        if config_path.exists():
            model_config = KogumaConfig.from_pretrained(checkpoint_path)
        else:
            raise ValueError(f"Model config not found at {config_path}")
    
    # Initialize model
    from koguma.models import KogumaForCausalLM
    model = KogumaForCausalLM(model_config)
    
    # Load state dict
    state_dict_path = checkpoint_path / "pytorch_model.bin"
    if state_dict_path.exists():
        state_dict = torch.load(state_dict_path, map_location=device)
        model.load_state_dict(state_dict, strict=strict)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    else:
        logger.warning(f"No checkpoint found at {state_dict_path}")
    
    # Load training state if exists
    training_state_path = checkpoint_path / "training_state.json"
    if training_state_path.exists():
        with open(training_state_path, "r") as f:
            training_state = json.load(f)
        logger.info(f"Loaded training state: {training_state}")
    
    return model


def save_optimizer_state(
    optimizer,
    output_dir: Path,
    accelerator: Optional[Accelerator] = None
):
    """Save optimizer state."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if accelerator is not None:
        accelerator.save(optimizer.state_dict(), output_dir / "optimizer.pt")
    else:
        torch.save(optimizer.state_dict(), output_dir / "optimizer.pt")
    
    logger.info(f"Optimizer state saved to {output_dir}")


def load_optimizer_state(
    optimizer,
    checkpoint_path: Path,
    device: str = "cuda"
):
    """Load optimizer state."""
    checkpoint_path = Path(checkpoint_path)
    optimizer_path = checkpoint_path / "optimizer.pt"
    
    if optimizer_path.exists():
        optimizer_state = torch.load(optimizer_path, map_location=device)
        optimizer.load_state_dict(optimizer_state)
        logger.info(f"Loaded optimizer state from {checkpoint_path}")
    else:
        logger.warning(f"No optimizer state found at {optimizer_path}")
    
    return optimizer