#!/bin/bash
# setup.sh - Unix/Mac Setup Script for pagent

set -e

echo "--- pagent Setup (Unix/Mac) ---"

# 1. Check Python Version
if ! command -v python3.11 &> /dev/null; then
    echo "Error: Python 3.11 is required but not found."
    echo "Please install it using your package manager (e.g., brew install python@3.11)"
    exit 1
fi

# 2. Create Virtual Environment
echo "Creating virtual environment (.venv)..."
python3.11 -m venv .venv

# 3. Install Dependencies
echo "Installing dependencies..."
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .

echo "--- Setup Complete ---"
echo "To activate the environment, run: source .venv/bin/activate"
echo "To start the platform, run: python -m agent_platform.cli"
