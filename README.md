# Video Subtitle Removal API

A microservice-based API for removing hardcoded subtitles from videos using AI.

## Project Structure

This project is organized as a monorepo with the following components:

- `api/`: FastAPI service for handling API requests
- `worker/`: GPU worker for video processing
- `shared/`: Shared models and utilities
- `infra/`: Infrastructure configuration (Docker, etc.)
- `scripts/`: Utility scripts

## Development Setup

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- NVIDIA GPU with CUDA support (for GPU worker)

### Installation

1. Clone the repository

```bash
git clone https://github.com/BrightOnAnalytics/vsr-api.git
cd vsr-api
```

2. Set up the development environment

```bash
make dev
```

3. Start the local development environment

```bash
make run
```

This will start the following services:
- MongoDB
- RabbitMQ
- MinIO (S3-compatible storage)
- API service
- Mock worker service

## API Endpoints

- `GET /healthz`: Health check endpoint
- Additional endpoints will be documented as they are implemented

## Architecture

The Video Subtitle Removal API uses the following technologies:

- **FastAPI**: Web framework for the API
- **PyTorch**: Deep learning framework for subtitle removal
- **MongoDB**: Database for storing job information
- **RabbitMQ**: Message queue for job processing
- **DigitalOcean Spaces (S3)**: Object storage for videos
- **Docker**: Containerization

## Development Workflow

1. Make changes to the code
2. Run linters: `make lint`
3. Run tests: `make test`
4. Start the local environment: `make run`

## Deployment

The application can be deployed using Docker Compose:

```bash
docker compose -f infra/docker-compose.prod.yml up -d
```

See the deployment documentation for more details.
