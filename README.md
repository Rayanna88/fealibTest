# pyfealib Test Environment

Python 3.10 project environment for testing with pyfealib on Linux x86_64.

## Requirements

- Python 3.10 (required - wheel is `cp310` specific)
- Linux x86_64 (manylinux_2_34 wheel)
- Or Docker for cross-platform support

## Files

- `pyfealib-1.0.0-cp310-cp310-manylinux_2_34_x86_64.whl` - pyfealib wheel (must be added)
- `Dockerfile` - Docker environment with Python 3.10
- `docker-compose.yml` - Docker Compose configuration
- `setup.sh` - Native Linux setup script
- `requirements.txt` - Python dependencies
- `test_installation.py` - Installation verification script
- `config.yaml` - Example YAML configuration

## Installation

### Option 1: Native Linux (x86_64 with Python 3.10)

```bash
pip install pyfealib-1.0.0-cp310-cp310-manylinux_2_34_x86_64.whl
pip install pytest pytest-xdist pyyaml
```

Or use the setup script:
```bash
./setup.sh
```

### Option 2: Docker (Recommended for macOS/Windows)

```bash
# Build the image
docker build -t fealib-env .

# Run the container
docker run -it fealib-env

# Or use docker-compose
docker-compose run --rm fealib
```

## Verify Installation

```bash
python test_installation.py
```

## Usage

### pytest (single process)
```bash
pytest -v
```

### pytest (parallel execution)
```bash
pytest -v -n auto
```

### Load YAML configuration in Python
```python
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

print(config)