#!/bin/bash
# Setup script for pyfealib environment
# Requires: Python 3.10 on Linux x86_64

set -e

# ── Logging setup ──────────────────────────────────────────────────────────────
LOG_FILE="setup_$(date '+%Y%m%d_%H%M%S').log"

# Print the mandatory header line into the log file first
echo "the log info while running the scripts in jump server" > "$LOG_FILE"

# From this point forward, everything printed to stdout/stderr is ALSO written
# to the log file (tee appends so the header line above is preserved).
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Log file: $LOG_FILE"
echo "──────────────────────────────────────────────────────────────────────────"
# ───────────────────────────────────────────────────────────────────────────────

echo "=========================================="
echo "pyfealib Environment Setup"
echo "=========================================="

# Check if running on Linux
if [[ "$(uname)" != "Linux" ]]; then
    echo "WARNING: This wheel file is for Linux x86_64 only."
    echo "Current OS: $(uname)"
    echo "Please use a Linux machine or Docker."
    echo ""
    echo "For Docker, run:"
    echo "  docker build -t fealib-env ."
    echo "  docker run -it fealib-env"
    exit 1
fi

# Check architecture
arch=$(uname -m)
if [[ "$arch" != "x86_64" ]]; then
    echo "WARNING: This wheel file is for x86_64 only."
    echo "Current architecture: $arch"
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
python_major=$(echo "$python_version" | cut -d'.' -f1)
python_minor=$(echo "$python_version" | cut -d'.' -f2)

echo "Python version: $python_version"
echo "Architecture: $arch"
echo ""

if [[ "$python_major" != "3" || "$python_minor" != "10" ]]; then
    echo "ERROR: Python 3.10 is required, but found $python_version"
    echo ""
    echo "To install Python 3.10:"
    echo "  Ubuntu/Debian: sudo apt install python3.10 python3.10-venv python3.10-dev"
    echo "  CentOS/RHEL:   sudo yum install python3.10"
    exit 1
fi

# Create virtual environment (recommended)
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install test dependencies
echo "Installing pytest and pytest-xdist..."
pip install pytest pytest-xdist

# Install PyYAML
echo "Installing PyYAML..."
pip install pyyaml

# Install pyfealib wheel
WHEEL_FILE="pyfealib-1.0.0-cp310-cp310-manylinux_2_34_x86_64.whl"
if [ -f "$WHEEL_FILE" ]; then
    echo "Installing pyfealib from $WHEEL_FILE..."
    pip install "$WHEEL_FILE"
else
    echo "WARNING: $WHEEL_FILE not found!"
    echo "Please download it and place it in this directory, then run:"
    echo "  pip install $WHEEL_FILE"
fi

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
python test_installation.py

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To use the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run tests (if any):"
echo "  pytest -v"
echo ""
echo "To run tests with parallel execution:"
echo "  pytest -v -n auto"