# setup.ps1 - Windows Setup Script for pagent

Write-Host "--- pagent Setup (Windows) ---" -ForegroundColor Cyan

# 1. Check Python Version
$python = "python"
$version = & $python --version 2>&1
if ($version -match "3.11") {
    Write-Host "Found Python 3.11"
} else {
    Write-Host "Error: Python 3.11 is required but not found in 'python' command." -ForegroundColor Red
    Write-Host "Please install it from python.org"
    exit 1
}

# 2. Create Virtual Environment
Write-Host "Creating virtual environment (.venv)..."
& $python -m venv .venv

# 3. Install Dependencies
Write-Host "Installing dependencies..."
& .\.venv\Scripts\pip install --upgrade pip
& .\.venv\Scripts\pip install -e .

Write-Host "--- Setup Complete ---" -ForegroundColor Green
Write-Host "To activate the environment, run: .\.venv\Scripts\Activate.ps1"
Write-Host "To start the platform, run: python -m agent_platform.cli"
