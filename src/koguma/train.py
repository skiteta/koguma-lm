"""Training script for Koguma-LM."""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import torch
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    get_scheduler,
    PreTrainedTokenizerFast
)

from koguma.models import KogumaForCausalLM, KogumaConfig
from koguma.data import KogumaDataset, create_data_collator
from koguma.utils.checkpoint import save_checkpoint, load_checkpoint

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Koguma-LM")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to model configuration file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )
    
    return parser.parse_args()


def load_configs(training_config_path: str, model_config_path: str):
    """Load configuration files."""
    with open(training_config_path, 'r') as f:
        training_config = yaml.safe_load(f)
    
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    
    return training_config, model_config


def setup_model_and_tokenizer(model_config: dict, resume_from: Optional[str] = None):
    """Initialize model and tokenizer."""
    # Load tokenizer
    tokenizer_path = model_config.get('tokenizer', {}).get('model_path')
    if tokenizer_path and Path(tokenizer_path).exists():
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    else:
        # Use a default tokenizer for now
        logger.warning("Tokenizer not found, using default tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-small")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
    config = KogumaConfig(**model_config['model'])
    
    if resume_from:
        model = load_checkpoint(resume_from, config)
        logger.info(f"Resumed from checkpoint: {resume_from}")
    else:
        model = KogumaForCausalLM(config)
        logger.info("Initialized new model")
    
    # Log model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return model, tokenizer


def train_epoch(
    model,
    train_loader,
    optimizer,
    lr_scheduler,
    accelerator,
    epoch,
    gradient_accumulation_steps
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    progress_bar = tqdm(
        total=len(train_loader),
        disable=not accelerator.is_local_main_process,
        desc=f"Epoch {epoch}"
    )
    
    for step, batch in enumerate(train_loader):
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.detach().float()
        
        # Logging
        if step % 10 == 0:
            progress_bar.set_postfix({
                'loss': loss.item(),
                'lr': optimizer.param_groups[0]['lr']
            })
        
        progress_bar.update(1)
    
    progress_bar.close()
    
    return total_loss.item() / len(train_loader)


def evaluate(model, eval_loader, accelerator):
    """Evaluate the model."""
    model.eval()
    losses = []
    
    for batch in tqdm(eval_loader, disable=not accelerator.is_local_main_process):
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            losses.append(accelerator.gather_for_metrics(loss))
    
    losses = torch.cat(losses)
    eval_loss = torch.mean(losses)
    
    try:
        perplexity = torch.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    
    return eval_loss.item(), perplexity.item()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configurations
    training_config, model_config = load_configs(args.config, args.model_config)
    train_cfg = training_config['training']
    data_cfg = training_config['data']
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=train_cfg['gradient_accumulation_steps'],
        mixed_precision='fp16' if train_cfg.get('fp16', False) else 'no',
        log_with="wandb" if args.wandb_project else None,
        project_config={"project_name": args.wandb_project} if args.wandb_project else None,
    )
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    # Set seed
    set_seed(args.seed)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_config, args.resume_from_checkpoint)
    
    # Setup datasets
    train_dataset = KogumaDataset(
        data_paths=data_cfg['pretrain_data'],
        tokenizer=tokenizer,
        block_size=data_cfg['block_size'],
        streaming=True  # Use streaming for large datasets
    )
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_cfg['per_device_train_batch_size'],
        shuffle=True,
        collate_fn=create_data_collator(tokenizer),
        num_workers=data_cfg.get('preprocessing_num_workers', 4),
        pin_memory=True
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg['learning_rate'],
        betas=(train_cfg['adam_beta1'], train_cfg['adam_beta2']),
        eps=train_cfg['adam_epsilon'],
        weight_decay=train_cfg['weight_decay']
    )
    
    # Setup learning rate scheduler
    num_training_steps = len(train_dataloader) * train_cfg['num_train_epochs']
    num_warmup_steps = int(train_cfg['warmup_ratio'] * num_training_steps)
    
    lr_scheduler = get_scheduler(
        name=train_cfg['lr_scheduler_type'],
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Prepare with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    # Training loop
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {train_cfg['num_train_epochs']}")
    logger.info(f"  Instantaneous batch size per device = {train_cfg['per_device_train_batch_size']}")
    logger.info(f"  Total train batch size = {train_cfg['per_device_train_batch_size'] * accelerator.num_processes * train_cfg['gradient_accumulation_steps']}")
    logger.info(f"  Gradient Accumulation steps = {train_cfg['gradient_accumulation_steps']}")
    logger.info(f"  Total optimization steps = {num_training_steps}")
    
    # Train
    best_loss = float('inf')
    
    for epoch in range(train_cfg['num_train_epochs']):
        logger.info(f"Starting epoch {epoch + 1}/{train_cfg['num_train_epochs']}")
        
        # Train
        avg_loss = train_epoch(
            model,
            train_dataloader,
            optimizer,
            lr_scheduler,
            accelerator,
            epoch + 1,
            train_cfg['gradient_accumulation_steps']
        )
        
        logger.info(f"Epoch {epoch + 1} - Average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if accelerator.is_main_process:
            if avg_loss < best_loss:
                best_loss = avg_loss
                output_dir = Path(args.output_dir) / f"checkpoint-best"
                save_checkpoint(
                    model,
                    tokenizer,
                    output_dir,
                    accelerator,
                    epoch=epoch + 1,
                    loss=avg_loss
                )
                logger.info(f"Saved best checkpoint to {output_dir}")
            
            # Save periodic checkpoint
            if (epoch + 1) % train_cfg.get('save_epochs', 1) == 0:
                output_dir = Path(args.output_dir) / f"checkpoint-epoch-{epoch + 1}"
                save_checkpoint(
                    model,
                    tokenizer,
                    output_dir,
                    accelerator,
                    epoch=epoch + 1,
                    loss=avg_loss
                )
    
    # Save final model
    if accelerator.is_main_process:
        output_dir = Path(args.output_dir) / "final"
        save_checkpoint(model, tokenizer, output_dir, accelerator)
        logger.info(f"Saved final model to {output_dir}")
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()