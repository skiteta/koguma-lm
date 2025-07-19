"""Knowledge distillation training script for Koguma-LM."""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional, Dict

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
from koguma.data import DistillationDataset, create_data_collator
from koguma.distill import MultiTeacherDistiller, DistillationDataGenerator
from koguma.utils.checkpoint import save_checkpoint, load_checkpoint

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Knowledge distillation for Koguma-LM")
    
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
        "--student_checkpoint",
        type=str,
        default=None,
        help="Path to pretrained student model checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/distillation",
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to distillation data (overrides config)"
    )
    parser.add_argument(
        "--generate_data",
        action="store_true",
        help="Generate distillation data before training"
    )
    parser.add_argument(
        "--num_samples_per_teacher",
        type=int,
        default=10000,
        help="Number of samples to generate per teacher"
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
    
    return parser.parse_args()


def generate_distillation_data(
    teacher_configs: list,
    output_dir: str,
    num_samples_per_teacher: int,
    device: str = "mps"  # Default to MPS for M3 Ultra
) -> str:
    """Generate distillation data using teacher models."""
    logger.info("Generating distillation data...")
    
    # Setup teacher models dict
    teacher_models = {config['name']: config['name'] for config in teacher_configs}
    
    # Initialize data generator
    generator = DistillationDataGenerator(
        teacher_models=teacher_models,
        output_dir=output_dir,
        device=device
    )
    
    # Generate data
    data_path = generator.generate_distillation_dataset(
        num_samples_per_teacher=num_samples_per_teacher,
        batch_size=4,  # Adjust based on available memory
        save_intermediate=True
    )
    
    logger.info(f"Generated distillation data saved to: {data_path}")
    return data_path


def distill_epoch(
    student_model,
    distiller: MultiTeacherDistiller,
    train_loader,
    optimizer,
    lr_scheduler,
    accelerator,
    epoch,
    config: Dict
):
    """Train student model for one epoch using distillation."""
    student_model.train()
    total_loss = 0
    total_distill_loss = 0
    total_task_loss = 0
    
    progress_bar = tqdm(
        total=len(train_loader),
        disable=not accelerator.is_local_main_process,
        desc=f"Distillation Epoch {epoch}"
    )
    
    for step, batch in enumerate(train_loader):
        with accelerator.accumulate(student_model):
            # Get task type based on batch metadata
            task_type = None
            if 'teacher' in batch:
                # Infer task type from teacher specialization
                teacher_name = batch['teacher'][0]  # Assuming uniform batch
                task_type = distiller.teacher_specializations.get(teacher_name, 'general')
            
            # Perform distillation step
            loss, metrics = distiller.adaptive_distillation_step(
                student_model,
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch.get('labels'),
                task_type=task_type
            )
            
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(student_model.parameters(), 1.0)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.detach().float()
        total_distill_loss += metrics.get('distill_loss', 0)
        total_task_loss += metrics.get('task_loss', 0)
        
        # Logging
        if step % 10 == 0:
            progress_bar.set_postfix({
                'loss': loss.item(),
                'distill': metrics.get('distill_loss', 0),
                'task': metrics.get('task_loss', 0),
                'lr': optimizer.param_groups[0]['lr']
            })
        
        progress_bar.update(1)
    
    progress_bar.close()
    
    avg_metrics = {
        'total_loss': total_loss.item() / len(train_loader),
        'distill_loss': total_distill_loss / len(train_loader),
        'task_loss': total_task_loss / len(train_loader)
    }
    
    return avg_metrics


def main():
    """Main distillation function."""
    args = parse_args()
    
    # Load configurations
    with open(args.config, 'r') as f:
        training_config = yaml.safe_load(f)
    
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)
    
    train_cfg = training_config['training']
    distill_cfg = training_config['distillation']
    data_cfg = training_config['data']
    teacher_configs = model_config['teacher_models']
    
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
    
    # Generate distillation data if requested
    if args.generate_data:
        data_path = generate_distillation_data(
            teacher_configs,
            data_cfg['distill_data_path'],
            args.num_samples_per_teacher,
            device="mps" if torch.backends.mps.is_available() else "cuda"
        )
    else:
        data_path = args.data_path or os.path.join(
            data_cfg['distill_data_path'],
            "distillation_dataset.jsonl"
        )
    
    # Setup tokenizer
    tokenizer_path = model_config.get('tokenizer', {}).get('model_path')
    if tokenizer_path and Path(tokenizer_path).exists():
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    else:
        logger.warning("Tokenizer not found, using default tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-small")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize student model
    config = KogumaConfig(**model_config['model'])
    
    if args.student_checkpoint:
        student_model = load_checkpoint(args.student_checkpoint, config)
        logger.info(f"Loaded student model from: {args.student_checkpoint}")
    else:
        student_model = KogumaForCausalLM(config)
        logger.info("Initialized new student model")
    
    # Initialize multi-teacher distiller
    distiller = MultiTeacherDistiller(
        student_model=student_model,
        teacher_configs=teacher_configs,
        temperature=distill_cfg['temperature'],
        alpha=distill_cfg['alpha'],
        device=accelerator.device
    )
    
    # Setup dataset
    train_dataset = DistillationDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        block_size=data_cfg['block_size'],
        include_teacher_logits=False  # We'll compute on-the-fly
    )
    
    # Create data loader
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
        student_model.parameters(),
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
    student_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        student_model, optimizer, train_dataloader, lr_scheduler
    )
    
    # Training loop
    logger.info("***** Running knowledge distillation *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {train_cfg['num_train_epochs']}")
    logger.info(f"  Instantaneous batch size per device = {train_cfg['per_device_train_batch_size']}")
    logger.info(f"  Total train batch size = {train_cfg['per_device_train_batch_size'] * accelerator.num_processes * train_cfg['gradient_accumulation_steps']}")
    logger.info(f"  Gradient Accumulation steps = {train_cfg['gradient_accumulation_steps']}")
    logger.info(f"  Total optimization steps = {num_training_steps}")
    logger.info(f"  Number of teachers = {len(teacher_configs)}")
    logger.info(f"  Distillation temperature = {distill_cfg['temperature']}")
    logger.info(f"  Distillation alpha = {distill_cfg['alpha']}")
    
    # Train
    best_loss = float('inf')
    
    for epoch in range(train_cfg['num_train_epochs']):
        logger.info(f"Starting distillation epoch {epoch + 1}/{train_cfg['num_train_epochs']}")
        
        # Train
        metrics = distill_epoch(
            student_model,
            distiller,
            train_dataloader,
            optimizer,
            lr_scheduler,
            accelerator,
            epoch + 1,
            distill_cfg
        )
        
        logger.info(
            f"Epoch {epoch + 1} - "
            f"Total loss: {metrics['total_loss']:.4f}, "
            f"Distill loss: {metrics['distill_loss']:.4f}, "
            f"Task loss: {metrics['task_loss']:.4f}"
        )
        
        # Save checkpoint
        if accelerator.is_main_process:
            if metrics['total_loss'] < best_loss:
                best_loss = metrics['total_loss']
                output_dir = Path(args.output_dir) / f"checkpoint-best"
                save_checkpoint(
                    student_model,
                    tokenizer,
                    output_dir,
                    accelerator,
                    epoch=epoch + 1,
                    loss=metrics['total_loss'],
                    metrics=metrics
                )
                logger.info(f"Saved best checkpoint to {output_dir}")
            
            # Save periodic checkpoint
            if (epoch + 1) % train_cfg.get('save_epochs', 1) == 0:
                output_dir = Path(args.output_dir) / f"checkpoint-epoch-{epoch + 1}"
                save_checkpoint(
                    student_model,
                    tokenizer,
                    output_dir,
                    accelerator,
                    epoch=epoch + 1,
                    loss=metrics['total_loss'],
                    metrics=metrics
                )
    
    # Save final model
    if accelerator.is_main_process:
        output_dir = Path(args.output_dir) / "final"
        save_checkpoint(student_model, tokenizer, output_dir, accelerator)
        logger.info(f"Saved final distilled model to {output_dir}")
    
    logger.info("Knowledge distillation completed!")


if __name__ == "__main__":
    main()