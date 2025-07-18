# Koguma-LM

A small Japanese language model (SLM) with multi-teacher knowledge distillation.

## Features

- 350M parameters optimized for Japanese
- Multi-teacher distillation from multiple expert models
- Optimized for long-context processing (8K-32K tokens)
- Commercial use friendly architecture
- No RLHF required - quality through data curation

## Setup

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

## Hardware Requirements

- Training: NVIDIA GPU with 24GB+ VRAM (tested on RTX A5500)
- Data Generation: Apple Silicon with 512GB unified memory (tested on M3 Ultra)
- Inference: 2-4GB RAM depending on quantization

## Project Structure

```
koguma-lm/
├── src/koguma/
│   ├── models/      # Model architectures
│   ├── data/        # Data processing utilities
│   ├── distill/     # Knowledge distillation
│   └── utils/       # Common utilities
├── configs/         # Configuration files
├── scripts/         # Training and evaluation scripts
├── notebooks/       # Jupyter notebooks for experiments
└── data/           # Data directory
```