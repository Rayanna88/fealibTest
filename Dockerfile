FROM python:3.10

# Install manylinux compatibility and x86_64 architecture support
RUN dpkg --add-architecture amd64 || true
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY . /workspace

RUN pip install --upgrade pip setuptools wheel

# Install required packages (pytest, pytest-xdist, pyyaml)
RUN pip install pytest pytest-xdist pyyaml

# Install pyfealib wheel (if it exists)
RUN if [ -f pyfealib-1.0.0-cp310-cp310-manylinux_2_34_x86_64.whl ]; then \
      pip install pyfealib-1.0.0-cp310-cp310-manylinux_2_34_x86_64.whl; \
    else \
      echo "Warning: pyfealib wheel file not found. Please place it in the project directory."; \
    fi

RUN python --version && pip --version

CMD ["/bin/bash"]