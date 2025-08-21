FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy shared package
COPY shared/ /app/shared/

# Copy API package
COPY api/ /app/api/

# Install shared package
RUN cd /app/shared && pip install --no-cache-dir -e .

# Install API package
RUN cd /app/api && pip install --no-cache-dir -e .

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Expose port
EXPOSE 8000

# Run the API
CMD ["python", "-m", "vsr_api.main"]
