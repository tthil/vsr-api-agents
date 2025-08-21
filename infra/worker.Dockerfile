# For local development on ARM64 (Apple Silicon), use CPU-only PyTorch
# For production GPU workloads, switch to: pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU version for ARM64 compatibility
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy shared package
COPY shared/ /app/shared/

# Copy worker package
COPY worker/ /app/worker/

# Install shared package
RUN cd /app/shared && pip install --no-cache-dir -e .

# Install only essential worker dependencies for local development
RUN pip install --no-cache-dir \
    pika==1.3.* \
    pymongo==4.9.* \
    motor==3.6.* \
    python-dotenv==1.0.1 \
    structlog==24.1.* \
    pydantic-settings==2.0.* \
    boto3==1.34.* \
    botocore==1.34.*

# Copy worker source code directly (skip heavy AI dependencies for local dev)
COPY worker/vsr_worker/ /app/vsr_worker/

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Run the working simple consumer
CMD ["python", "-m", "vsr_worker.queue.simple_consumer"]
