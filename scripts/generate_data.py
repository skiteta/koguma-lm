#!/usr/bin/env python3
"""Script to generate distillation data using M3 Ultra."""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
from koguma.distill import DistillationDataGenerator

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate distillation data")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to model configuration file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/distillation",
        help="Output directory for generated data"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of samples to generate per teacher"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for generation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device to use (mps/cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        model_config = yaml.safe_load(f)
    
    teacher_configs = model_config['teacher_models']
    
    # Setup teacher models dict
    teacher_models = {
        config['name']: config['name'] 
        for config in teacher_configs
    }
    
    logger.info(f"Generating data from {len(teacher_models)} teacher models")
    logger.info(f"Teachers: {list(teacher_models.keys())}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Samples per teacher: {args.num_samples}")
    
    # Check device availability
    if args.device == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS not available, falling back to CPU")
        args.device = "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    logger.info(f"Using device: {args.device}")
    
    # Initialize generator
    generator = DistillationDataGenerator(
        teacher_models=teacher_models,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Generate data
    try:
        data_path = generator.generate_distillation_dataset(
            num_samples_per_teacher=args.num_samples,
            batch_size=args.batch_size,
            save_intermediate=True
        )
        
        logger.info(f"Successfully generated data at: {data_path}")
        
        # Print statistics
        import json
        num_lines = 0
        with open(data_path, 'r') as f:
            for line in f:
                num_lines += 1
        
        logger.info(f"Total samples generated: {num_lines}")
        
    except Exception as e:
        logger.error(f"Error generating data: {e}")
        raise


if __name__ == "__main__":
    main()