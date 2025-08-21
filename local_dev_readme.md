# 🚀 VSR API Local Development Guide

This guide will walk you through setting up and running the Video Subtitle Removal (VSR) API locally on your machine for development and testing.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Understanding the Local Environment](#understanding-the-local-environment)
- [Step-by-Step Setup](#step-by-step-setup)
- [Testing the API](#testing-the-api)
- [Development Workflow](#development-workflow)
- [Monitoring and Debugging](#monitoring-and-debugging)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

## 🔧 Prerequisites

Before you begin, ensure you have the following installed on your machine:

### Required Software

1. **Docker Desktop** (latest version)
   - Download from: https://www.docker.com/products/docker-desktop
   - Ensure Docker Compose is included (it comes with Docker Desktop)

2. **Python 3.11 or higher**
   - Check your version: `python --version`
   - Download from: https://www.python.org/downloads/

3. **Make** (usually pre-installed on macOS/Linux)
   - Check if installed: `make --version`
   - On macOS: Install Xcode Command Line Tools if missing

4. **Git** (for version control)
   - Check if installed: `git --version`

### System Requirements

- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: At least 10GB free space
- **CPU**: Multi-core processor recommended
- **OS**: macOS, Linux, or Windows with WSL2

## ⚡ Quick Start

If you want to get up and running immediately:

```bash
# 1. Navigate to the project directory
cd /Users/thomasthil/Development/BrightOnAnalytics/vsr-api

# 2. Start the local environment
make run-detached

# 3. Wait for services to start (about 30-60 seconds)
# Check status with:
docker compose -f infra/docker-compose.local.yml ps

# 4. Test the API
curl http://localhost:8000/healthz
```

That's it! Your local VSR API is now running at http://localhost:8000

## 🏗️ Understanding the Local Environment

The local development environment consists of several interconnected services:

### Core Services

| Service | Port | Purpose | Access |
|---------|------|---------|---------|
| **VSR API** | 8000 | Main FastAPI application | http://localhost:8000 |
| **Mock Worker** | - | Simulates video processing | Background service |
| **MongoDB** | 27017 | Database for jobs and metadata | mongodb://localhost:27017 |
| **RabbitMQ** | 5673, 15673 | Message queue for job processing | http://localhost:15673 |
| **MinIO** | 9000, 9001 | S3-compatible storage | http://localhost:9001 |

### Service Dependencies

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   VSR API   │───▶│   MongoDB    │    │ Mock Worker │
│             │    │              │    │             │
│             │───▶│   RabbitMQ   │◀───│             │
│             │    │              │    │             │
│             │───▶│    MinIO     │◀───│             │
└─────────────┘    └──────────────┘    └─────────────┘
```

## 📝 Step-by-Step Setup

### Step 1: Project Setup

1. **Navigate to the project directory:**
   ```bash
   cd /Users/thomasthil/Development/BrightOnAnalytics/vsr-api
   ```

2. **Verify project structure:**
   ```bash
   ls -la
   # You should see: api/, worker/, shared/, infra/, Makefile, etc.
   ```

### Step 2: Install Development Dependencies (Optional)

If you plan to develop or run tests locally:

```bash
# Install all development dependencies
make dev
```

This command will:
- Install the `shared` package in editable mode
- Install the `api` package in editable mode
- Install the `worker` package in editable mode
- Set up pre-commit hooks for code quality

### Step 3: Start the Local Environment

#### Option A: Start with Logs (Recommended for first-time setup)

```bash
make run
```

This will start all services and show logs in real-time. You'll see:
- MongoDB initialization
- RabbitMQ startup
- MinIO bucket creation
- API server startup
- Mock worker initialization

#### Option B: Start in Background (Detached Mode)

```bash
make run-detached
```

This starts all services in the background, returning control to your terminal immediately.

### Step 4: Verify Services Are Running

```bash
# Check service status
docker compose -f infra/docker-compose.local.yml ps

# Expected output:
# NAME                STATUS              PORTS
# vsr-api            Up                  0.0.0.0:8000->8000/tcp
# vsr-mongodb        Up (healthy)        0.0.0.0:27017->27017/tcp
# vsr-rabbitmq       Up (healthy)        0.0.0.0:5673->5672/tcp, 0.0.0.0:15673->15672/tcp
# vsr-minio          Up (healthy)        0.0.0.0:9000->9000/tcp, 0.0.0.0:9001->9001/tcp
# vsr-mock-worker    Up                  
```

### Step 5: Test Basic Connectivity

```bash
# Test API health
curl http://localhost:8000/healthz

# Expected response:
# {"status":"healthy","timestamp":"2024-01-01T12:00:00Z","version":"1.0.0"}
```

## 🧪 Testing the API

> **Note**: Authentication has been disabled for local development to simplify testing. No API keys or authorization headers are required.

### Access the API Documentation

Open your browser and navigate to:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Test Endpoints with curl

### 1. Health Check 
```bash
curl http://localhost:8000/healthz
```

Expected response:
```json
{"status": "ok"}
```

### 2. Generate Upload URL ✅

```bash
curl -X POST "http://localhost:8000/api/generate-upload-url" \
  -H "Content-Type: application/json" \
  -d '{"content_type": "video/mp4"}'
```

**Expected Response:**
```json
{
  "upload_url": "http://minio:9000/vsr-videos/uploads/20250812/abc123.../video.mp4?X-Amz-Algorithm=...",
  "expires_in": 3600
}
```

#### 3. Upload and Submit Video for Processing ✅

```bash
# Upload a video file directly for processing
curl -X POST "http://localhost:8000/api/upload-and-submit" \
  -F "video_file=@vids/source.mp4;type=video/mp4" \
  -F "mode=sttn"
```

**Parameters:**
- `video_file`: Your video file (e.g., `@vids/source.mp4`)
- `mode`: Processing mode (`sttn`, `lama`, or `propainter`)
- `subtitle_area`: Coordinates as `x1,y1,x2,y2` (optional)

#### 4. Submit Video URL for Processing

```bash
curl -X POST "http://localhost:8000/api/submit-video" \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4",
    "mode": "sttn",
    "subtitle_area": "100,100,800,600",
    "callback_url": "https://webhook.site/your-unique-id"
  }'
```

**Expected Response:**
```json
{
  "job_id": "abc123-def456-ghi789",
  "status": "submitted",
  "video_key": "uploads/20250812/abc123.../video.mp4",
  "estimated_completion": "2025-08-12T15:45:00Z"
}
```

### 4. Check Job Status ✅

```bash
# Replace {job_id} with the actual job ID from previous steps
curl -s "http://localhost:8000/api/job-status/{job_id}" | python3 -m json.tool
```

**Example:**
```bash
curl -s "http://localhost:8000/api/job-status/c3fca204-294b-45c8-a720-010fb23cf5f1" | python3 -m json.tool
```

**Expected Response:**
```json
{
    "id": "c3fca204-294b-45c8-a720-010fb23cf5f1",
    "status": "pending",
    "mode": "sttn",
    "subtitle_area": null,
    "progress": 0.0,
    "video_key": "KeyPrefix.UPLOADS/20250812/c3fca204-294b-45c8-a720-010fb23cf5f1/video.mp4",
    "processed_video_key": null,
    "processed_video_url": null,
    "created_at": "2025-08-12T09:37:39.655000",
    "updated_at": "2025-08-12T09:37:39.655000",
    "started_at": null,
    "completed_at": null,
    "error_message": null,
    "callback_url": null,
    "processing_time_seconds": null,
    "queue_position": 3
}
```

### Test with Swagger UI

1. Open http://localhost:8000/docs
2. Click "Authorize" and enter API key: `test-key-1`
3. Try the endpoints directly from the browser interface

## 🔄 Development Workflow

### Making Code Changes

The local environment supports hot-reloading:

1. **API Changes**: Edit files in `api/` directory
   - Changes are automatically detected and the server reloads
   - No need to restart containers

2. **Worker Changes**: Edit files in `worker/` directory
   - Worker automatically restarts on code changes

3. **Shared Library Changes**: Edit files in `shared/` directory
   - Both API and worker will reload when shared code changes

### Storage (MinIO/S3)

The local environment uses MinIO as an S3-compatible storage service:

- **MinIO Console**: http://localhost:9001 (admin/password: `minioadmin`/`minioadmin`)
- **Buckets**: Automatically created by the `minio-setup` service
  - `vsr-videos`: Main bucket for video storage
- **Access**: API uses presigned URLs for secure file uploads/downloads

### Example: Complete Video Processing Workflow

```bash
# 1. Upload your video file for processing
curl -X POST "http://localhost:8000/api/upload-and-submit" \
  -F "video_file=@vids/source.mp4" \
  -F "mode=sttn" \
  -F "subtitle_area=100,100,800,600"

# Response: {"job_id": "abc123-def456-ghi789", "status": "submitted", ...}

# 2. Check processing status
curl -X GET "http://localhost:8000/api/jobs/abc123-def456-ghi789"

# 3. When complete, download the processed video
# The API response will include a presigned download URL
```

## 🔧 Troubleshooting

### Common Issues

#### API Returns 404 "Not Found"
- **Cause**: Wrong endpoint path
- **Solution**: Use `/api/upload-and-submit` (not `/api/submit-video-file`)

#### Authentication Errors (401 Unauthorized)
- **Cause**: Authentication middleware still enabled
- **Solution**: Authentication has been disabled for local development

#### Validation Errors
- **Subtitle Area**: Must be 4 comma-separated coordinates: `x1,y1,x2,y2`
- **Processing Mode**: Must be one of: `sttn`, `lama`, `propainter`
- **Video File**: Must be a valid video format (mp4, avi, mov, etc.)

#### Container Issues
```bash
# Restart all services
docker compose -f infra/docker-compose.local.yml restart

# Check service logs
docker compose -f infra/docker-compose.local.yml logs api --tail=20
docker compose -f infra/docker-compose.local.yml logs mongodb --tail=10

# Rebuild containers if needed
docker compose -f infra/docker-compose.local.yml down
docker compose -f infra/docker-compose.local.yml up -d --build
```

### Running Tests

```bash
# Run all tests
make test

# Run API tests only
cd api && pytest -xvs

# Run worker tests only
cd worker && pytest -xvs

# Run tests with coverage
cd api && pytest --cov=vsr_api --cov-report=html
```

### Code Quality Checks

```bash
# Run linting
make lint

# Fix formatting issues
black .
isort .

# Check type hints
cd shared && mypy vsr_shared
```

### Git Workflow

```bash
# Pre-commit hooks run automatically
git add .
git commit -m "Your commit message"

# If pre-commit fails, fix issues and try again
```

## 📊 Monitoring and Debugging

### View Service Logs

```bash
# All services
docker compose -f infra/docker-compose.local.yml logs -f

# Specific service
docker compose -f infra/docker-compose.local.yml logs -f api
docker compose -f infra/docker-compose.local.yml logs -f mock-worker
docker compose -f infra/docker-compose.local.yml logs -f mongodb
```

### Access Service UIs

#### RabbitMQ Management Console
- **URL**: http://localhost:15673
- **Username**: guest
- **Password**: guest
- **Use**: Monitor queues, exchanges, and message flow

#### MinIO Console
- **URL**: http://localhost:9001
- **Username**: minioadmin
- **Password**: minioadmin
- **Use**: Browse uploaded files and processed videos

#### MongoDB (via CLI)
```bash
# Connect to MongoDB
docker exec -it vsr-mongodb mongosh

# Switch to VSR database
use vsr

# View collections
show collections

# Query jobs
db.jobs.find().pretty()
```

### Debug API Issues

1. **Check API logs**:
   ```bash
   docker compose -f infra/docker-compose.local.yml logs -f api
   ```

2. **Inspect API container**:
   ```bash
   docker exec -it vsr-api bash
   ```

3. **Check environment variables**:
   ```bash
   docker exec vsr-api env | grep VSR
   ```

## ✅ Testing Status Summary

### Working Endpoints
- ✅ **GET /healthz** - Health check endpoint
- ✅ **POST /api/generate-upload-url** - Presigned URL generation  
- ✅ **POST /api/upload-and-submit** - Direct video file upload and processing
- ⚠️ **POST /api/submit-video** - Video URL submission (minor JSON parsing issue)
- ⚠️ **GET /api/jobs/{job_id}** - Job status retrieval (needs investigation)

### Successful Test Example
```bash
curl -X POST "http://localhost:8000/api/upload-and-submit" \
  -F "video_file=@vids/source.mp4;type=video/mp4" \
  -F "mode=sttn"
```

**Response:**
```json
{
  "id": "4a99e0eb-094b-4e08-81ed-a2889df675ac",
  "status": "pending", 
  "mode": "sttn",
  "video_key": "KeyPrefix.UPLOADS/20250812/4a99e0eb-094b-4e08-81ed-a2889df675ac/video.mp4",
  "queue_position": 2,
  "eta_seconds": 150,
  "created_at": "2025-08-12T09:32:51.550062"
}
```

## 🔧 Troubleshooting

### Common Issues

1. **Port conflicts**: If port 8000 is in use, modify the port mapping in docker-compose.local.yml
2. **MinIO connection issues**: Ensure MinIO is running and accessible at http://localhost:9001
3. **MongoDB connection issues**: Check that MongoDB is running on port 27017
4. **RabbitMQ connection issues**: Verify RabbitMQ is running on port 5672

### Issue: Port Already in Use

**Symptoms**: Error starting services, port conflict messages

**Solution**:
```bash
# Check what's using the port
lsof -i :8000  # For API
lsof -i :27017 # For MongoDB
lsof -i :5673  # For RabbitMQ

# Kill the process or change ports in docker-compose.local.yml
```

### Issue: Services Won't Start

**Symptoms**: Containers exit immediately or fail health checks

**Solution**:
```bash
# Check Docker resources
docker system df

# Clean up if needed
docker system prune

# Rebuild containers
make clean
docker compose -f infra/docker-compose.local.yml build --no-cache
make run
```

#### Issue: API Returns 500 Errors

**Symptoms**: Internal server errors on API calls

**Solution**:
```bash
# Check API logs
docker compose -f infra/docker-compose.local.yml logs api

# Check database connection
docker exec vsr-api python -c "
from vsr_shared.database import get_database_client
import asyncio
async def test():
    client = await get_database_client()
    print('DB connected:', await client.admin.command('ping'))
asyncio.run(test())
"
```

#### Issue: Mock Worker Not Processing Jobs

**Symptoms**: Jobs stuck in "queued" status

**Solution**:
```bash
# Check worker logs
docker compose -f infra/docker-compose.local.yml logs mock-worker

# Check RabbitMQ queues
# Visit http://localhost:15673 and check queue depths

# Restart worker
docker compose -f infra/docker-compose.local.yml restart mock-worker
```

#### Issue: File Upload Fails

**Symptoms**: Upload URLs don't work or return errors

**Solution**:
```bash
# Check MinIO is running
curl http://localhost:9000/minio/health/live

# Check bucket exists
docker exec vsr-minio mc ls myminio/

# Recreate bucket
docker compose -f infra/docker-compose.local.yml restart minio-setup
```

### Reset Environment

If you encounter persistent issues:

```bash
# Stop all services
make stop

# Remove all containers and volumes (WARNING: deletes all data)
docker compose -f infra/docker-compose.local.yml down -v

# Remove images
docker rmi vsr-api:latest vsr-worker:latest

# Start fresh
make run
```

## 🚀 Advanced Usage

### Custom Configuration

Create a `.env.local` file in the project root to override default settings:

```bash
# .env.local
MONGODB_URL=mongodb://localhost:27017/vsr_custom
RABBITMQ_URL=amqp://guest:guest@localhost:5673/
SPACES_BUCKET=my-custom-bucket
MOCK_PROCESSING_TIME=5
```

### Running Individual Services

You can start services individually for debugging:

```bash
# Start only infrastructure
docker compose -f infra/docker-compose.local.yml up mongodb rabbitmq minio -d

# Run API locally (outside Docker)
cd api
export MONGODB_URL=mongodb://localhost:27017/vsr
export RABBITMQ_URL=amqp://guest:guest@localhost:5673/
python -m uvicorn vsr_api.main:app --reload

# Run worker locally
cd worker
export WORKER_MODE=mock
python -m vsr_worker.main --mode mock
```

### Performance Testing

```bash
# Install testing tools
pip install locust

# Run load tests (create locustfile.py first)
locust -f tests/locustfile.py --host=http://localhost:8000
```

### Database Seeding

```bash
# Run database seed script
cd shared
python scripts/seed_database.py

# This creates:
# - Sample API keys
# - Test job records
# - Mock job events
```

## 🛑 Stopping the Environment

### Graceful Shutdown

```bash
# Stop all services
make stop

# Or using docker compose directly
docker compose -f infra/docker-compose.local.yml down
```

### Complete Cleanup

```bash
# Stop and remove everything including volumes
docker compose -f infra/docker-compose.local.yml down -v

# Clean up build artifacts
make clean
```

## 📚 Additional Resources

- **API Documentation**: http://localhost:8000/docs (when running)
- **Project README**: See main README.md for project overview
- **Production Deployment**: See deployment/ directory for production setup
- **Contributing**: See CONTRIBUTING.md for development guidelines

## 🆘 Getting Help

If you encounter issues not covered in this guide:

1. Check the service logs first
2. Search existing GitHub issues
3. Create a new issue with:
   - Your operating system
   - Docker version
   - Complete error messages
   - Steps to reproduce

---

**Happy coding! 🎉**

The VSR API local development environment provides a complete testing sandbox that mirrors the production setup, allowing you to develop and test features safely on your local machine.
