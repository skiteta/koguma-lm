#!/bin/bash

# Setup script for Koguma-LM

echo "Setting up Koguma-LM development environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Create virtual environment
echo "Creating virtual environment..."
uv venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
uv pip install -e ".[dev]"

# Create necessary directories
echo "Creating project directories..."
mkdir -p outputs logs models checkpoints

echo "Setup complete! To activate the environment, run:"
echo "  source .venv/bin/activate"