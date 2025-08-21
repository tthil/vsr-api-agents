# Product Requirements Document (PRD)
## Video Subtitle Remover (VSR) API Service

**Version:** 1.0  
**Date:** August 2025  
**Status:** Ready for Implementation

---

## 1. Executive Summary

### 1.1 Purpose
Develop an API service that removes hardcoded subtitles from videos using AI technology, deployed on DigitalOcean infrastructure, with a focus on simplicity, reliability, and cost-effectiveness.

### 1.2 Key Metrics
- **Volume:** 10-15 videos per day
- **Video Specifications:** HD quality, 30-60 seconds duration
- **Processing Time:** 2-5 minutes per video
- **Budget:** $200-300 USD/month (actual: ~$60-70)
- **Availability:** Business hours processing with real-time notifications

---

## 2. Business Requirements

### 2.1 Problem Statement
Internal teams need to remove hardcoded subtitles from HD videos quickly and reliably without manual intervention or specialized software knowledge.

### 2.2 Success Criteria
- ✅ Videos processed within 5 minutes
- ✅ Real-time notifications on completion/failure
- ✅ Simple API integration with existing tools
- ✅ Monthly costs under $300
- ✅ 99% processing success rate for standard HD videos

### 2.3 Out of Scope
- Real-time/live video processing
- 4K or higher resolution videos
- Videos longer than 60 seconds
- Automatic retry on failures
- Multiple concurrent processing

---

## 3. Technical Architecture

### 3.1 System Overview
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Internal Tools  │────▶│ DO Spaces       │────▶│ API Server      │
│ (API Client)    │     │ (S3 Storage)    │     │ (CPU Droplet)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
                                                  ┌─────────────────┐
                                                  │ Message Queue   │
                                                  │ (RabbitMQ)      │
                                                  └─────────────────┘
                                                          │
                                                          ▼
                                                  ┌─────────────────┐
                                                  │ GPU Worker      │
                                                  │ (H100 Droplet)  │
                                                  └─────────────────┘
```

### 3.2 Technology Stack
- **API Framework:** FastAPI (Python 3.11)
- **Message Queue:** RabbitMQ
- **Database:** MongoDB
- **Storage:** DigitalOcean Spaces (S3-compatible)
- **GPU Processing:** NVIDIA H100 (80GB VRAM)
- **Container:** Docker with NVIDIA runtime
- **AI Models:** VSR (STTN, LAMA, PROPAINTER)

### 3.3 Infrastructure
- **API Server:** DigitalOcean CPU Droplet (4 vCPU, 8GB RAM)
- **GPU Worker:** DigitalOcean GPU Droplet (H100, 80GB VRAM)
- **Region:** NYC2 (GPU), NYC3 (API/Storage)
- **Networking:** 10 Gbps public, 25 Gbps private

---

## 4. API Specification

### 4.1 Authentication
- **Method:** Bearer token (API Key)
- **Header:** `Authorization: Bearer <API_KEY>`

### 4.2 Endpoints

#### POST /api/submit-video
Submit a video for subtitle removal.

**Request Option 1 - URL:**
```json
{
  "video_url": "https://example.com/video.mp4",
  "mode": "STTN",
  "subtitle_area": [100, 500, 1820, 600],
  "callback_url": "https://your-app.com/webhook"
}
```

**Request Option 2 - Pre-uploaded to Spaces:**
```json
{
  "video_key": "uploads/20250811/550e8400/video.mp4",
  "mode": "STTN",
  "subtitle_area": [100, 500, 1820, 600],
  "callback_url": "https://your-app.com/webhook"
}
```

#### POST /api/upload-and-submit
Upload a local file and submit for processing in one request.

**Request:** `multipart/form-data`
- `file`: The video file (binary)
- `mode`: Processing mode (STTN/LAMA/PROPAINTER)
- `subtitle_area`: Optional JSON array [x1, y1, x2, y2]
- `callback_url`: Webhook URL for notifications

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/upload-and-submit \
  -H "Authorization: Bearer local-dev-key" \
  -F "file=@/path/to/video.mp4" \
  -F "mode=STTN" \
  -F "callback_url=http://localhost:8001/webhook"
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Video uploaded and queued for processing",
  "estimated_completion_minutes": 3.0
}
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Video queued for processing. Position in queue: 2",
  "estimated_completion_minutes": 4.0
}
```

#### GET /api/job-status/{job_id}
Check the status of a processing job.

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress": 100.0,
  "created_at": "2025-08-11T10:30:00Z",
  "completed_at": "2025-08-11T10:32:45Z",
  "video_url": "https://nyc3.digitaloceanspaces.com/vsr-videos/processed/...",
  "processing_time_seconds": 165
}
```

#### GET /api/generate-upload-url
Get a pre-signed URL for direct video upload.

**Response:**
```json
{
  "upload_url": "https://nyc3.digitaloceanspaces.com/vsr-videos/...",
  "key": "uploads/20250811/550e8400/video.mp4",
  "expires_in": 3600
}
```

### 4.3 Webhook Notification
**Success Payload:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "video_url": "https://nyc3.digitaloceanspaces.com/vsr-videos/processed/...",
  "processing_time_seconds": 165
}
```

**Failure Payload:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "failed",
  "error_message": "GPU out of memory for PROPAINTER mode"
}
```

---

## 5. Local Development Environment

### 5.1 Mac Studio (ARM64) Setup

Since Mac doesn't have NVIDIA GPUs, we'll use a mock processing mode for local development.

#### Prerequisites
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required tools
brew install python@3.11 mongodb-community rabbitmq ffmpeg
brew install --cask docker

# Install MinIO (local S3 replacement)
brew install minio/stable/minio
```

#### Local Development Structure
```
vsr-api-local/
├── docker-compose.local.yml
├── .env.local
├── api/
│   ├── api_server_simple.py
│   ├── requirements.txt
│   └── Dockerfile.local
├── worker/
│   ├── worker_mock.py      # Mock worker for local dev
│   ├── requirements.txt
│   └── Dockerfile.local
└── scripts/
    ├── setup-local.sh
    └── test-api.py
```

#### docker-compose.local.yml
```yaml
version: '3.8'

services:
  # RabbitMQ
  rabbitmq:
    image: rabbitmq:3.12-management
    ports:
      - "5672:5672"
      - "15672:15672"
    environment:
      RABBITMQ_DEFAULT_USER: admin
      RABBITMQ_DEFAULT_PASS: localpass

  # MongoDB
  mongodb:
    image: mongo:7
    ports:
      - "27017:27017"
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD: localpass

  # MinIO (Local S3)
  minio:
    image: minio/minio
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    command: server /data --console-address ":9001"
    volumes:
      - minio-data:/data

  # API Server
  vsr-api:
    build:
      context: ./api
      dockerfile: Dockerfile.local
    ports:
      - "8000:8000"
    environment:
      - API_KEYS=local-dev-key
      - RABBITMQ_URL=amqp://admin:localpass@rabbitmq:5672/
      - MONGODB_URL=mongodb://admin:localpass@mongodb:27017/vsr?authSource=admin
      - SPACES_ENDPOINT=http://minio:9000
      - SPACES_KEY=minioadmin
      - SPACES_SECRET=minioadmin
      - SPACES_BUCKET=vsr-videos
      - LOCAL_DEV=true
    depends_on:
      - rabbitmq
      - mongodb
      - minio

  # Mock Worker (no GPU required)
  vsr-worker:
    build:
      context: ./worker
      dockerfile: Dockerfile.local
    environment:
      - RABBITMQ_URL=amqp://admin:localpass@rabbitmq:5672/
      - MONGODB_URL=mongodb://admin:localpass@mongodb:27017/vsr?authSource=admin
      - SPACES_ENDPOINT=http://minio:9000
      - SPACES_KEY=minioadmin
      - SPACES_SECRET=minioadmin
      - SPACES_BUCKET=vsr-videos
      - MOCK_MODE=true
    depends_on:
      - rabbitmq
      - mongodb
      - minio

volumes:
  minio-data:
```

#### Mock Worker Implementation
```python
# worker/worker_mock.py
import os
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MockVSRWorker:
    """Mock worker for local development without GPU"""
    
    async def process_job(self, job_id: str):
        """Simulate video processing"""
        logger.info(f"Mock processing job {job_id}")
        
        # Simulate processing stages
        stages = [
            (10, "Downloading video"),
            (30, "Loading AI model (mock)"),
            (50, "Analyzing video"),
            (70, "Removing subtitles (mock)"),
            (90, "Uploading result"),
            (100, "Complete")
        ]
        
        for progress, message in stages:
            await self.update_progress(job_id, progress, message)
            await asyncio.sleep(2)  # Simulate work
        
        # In mock mode, just copy input to output
        # Real implementation would process the video
        logger.info(f"Mock processing complete for {job_id}")
```

#### Local Development Workflow

1. **Start Services:**
```bash
# Clone the repository
git clone <your-repo>
cd vsr-api

# Start local environment
docker-compose -f docker-compose.local.yml up
```

2. **Create MinIO Bucket:**
```bash
# Access MinIO at http://localhost:9001
# Login: minioadmin/minioadmin
# Create bucket: vsr-videos
```

3. **Test API:**
```python
# scripts/test-api.py
import requests
import os

API_URL = "http://localhost:8000"
API_KEY = "local-dev-key"

# Test 1: Upload local file and process
def test_local_file_upload():
    video_path = "/Users/you/Desktop/test-video.mp4"
    
    with open(video_path, 'rb') as f:
        files = {'file': ('test-video.mp4', f, 'video/mp4')}
        data = {
            'mode': 'STTN',
            'callback_url': 'http://localhost:8001/webhook'
        }
        headers = {'Authorization': f'Bearer {API_KEY}'}
        
        response = requests.post(
            f"{API_URL}/api/upload-and-submit",
            files=files,
            data=data,
            headers=headers
        )
    
    print("Upload response:", response.json())
    return response.json()['job_id']

# Test 2: Submit video from URL
def test_url_submission():
    response = requests.post(
        f"{API_URL}/api/submit-video",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "video_url": "https://example.com/test.mp4",
            "mode": "STTN",
            "callback_url": "http://localhost:8001/webhook"
        }
    )
    print("URL submission response:", response.json())

# Test 3: Check job status
def check_job_status(job_id):
    response = requests.get(
        f"{API_URL}/api/job-status/{job_id}",
        headers={"Authorization": f"Bearer {API_KEY}"}
    )
    print("Job status:", response.json())

# Run tests
if __name__ == "__main__":
    # Test with local file
    job_id = test_local_file_upload()
    
    # Wait a bit and check status
    import time
    time.sleep(5)
    check_job_status(job_id)
```

#### Simple Webhook Server for Testing
```python
# scripts/webhook-server.py
from flask import Flask, request
import json

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    print("\n🔔 Webhook received!")
    print(json.dumps(data, indent=2))
    
    if data['status'] == 'completed':
        print(f"✅ Video ready at: {data['video_url']}")
    else:
        print(f"❌ Processing failed: {data['error_message']}")
    
    return "OK", 200

if __name__ == '__main__':
    print("Webhook server listening on http://localhost:8001/webhook")
    app.run(port=8001)
```

### 5.2 Development Best Practices

1. **Code Structure:**
   - Keep production and mock implementations separate
   - Use environment variables for configuration
   - Implement comprehensive logging

2. **Testing Strategy:**
   - Unit tests for API endpoints
   - Integration tests with mock worker
   - Load tests to simulate daily volume

3. **Debugging:**
   - Use VS Code with Python debugger
   - Monitor RabbitMQ management UI
   - Check MongoDB with Compass

---

## 6. Deployment Process

### 6.1 Initial Setup
1. Create DigitalOcean resources (Spaces, Droplets)
2. Configure DNS and SSL certificates
3. Deploy API server with Docker Compose
4. Deploy GPU worker on GPU droplet
5. Run smoke tests

### 6.2 CI/CD Pipeline
```yaml
# .github/workflows/deploy.yml
name: Deploy VSR API

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Deploy to DigitalOcean
        env:
          DO_API_TOKEN: ${{ secrets.DO_API_TOKEN }}
        run: |
          # Build and push Docker images
          docker build -t registry.digitalocean.com/vsr/api:latest ./api
          docker push registry.digitalocean.com/vsr/api:latest
          
          # Deploy via SSH
          ssh deploy@api-server "cd /opt/vsr && docker-compose pull && docker-compose up -d"
```

### 6.3 Monitoring & Alerts
- **Uptime Monitoring:** DigitalOcean Monitoring
- **Queue Monitoring:** RabbitMQ Management UI
- **Error Tracking:** MongoDB logs + Webhook failures
- **Cost Tracking:** DigitalOcean billing alerts

---

## 7. Operations & Maintenance

### 7.1 Daily Operations
- Monitor queue depth (should be 0-3 most of the time)
- Check for failed jobs in MongoDB
- Verify GPU worker health

### 7.2 Troubleshooting Guide

| Issue | Solution |
|-------|----------|
| GPU out of memory | Restart worker container |
| Queue backing up | Check worker logs, restart if needed |
| Slow processing | Verify video is HD and <60 seconds |
| Webhook failures | Check target URL accessibility |

### 7.3 Scaling Considerations
Current design handles 10-15 videos/day. To scale:
- Add more GPU workers (horizontal scaling)
- Implement job priority queues
- Consider batch processing during off-peak

---

## 8. Security & Compliance

### 8.1 Security Measures
- API key authentication
- HTTPS for all endpoints
- Private networking between services
- No storage of sensitive data

### 8.2 Data Retention
- Input videos: 7 days
- Processed videos: 30 days
- Job metadata: 90 days
- Automatic cleanup via Spaces lifecycle

---

## 9. Cost Analysis

### 9.1 Monthly Breakdown
| Component | Cost |
|-----------|------|
| GPU Droplet (45 min/day) | $35 |
| CPU Droplet (24/7) | $24 |
| DigitalOcean Spaces | $5 |
| Bandwidth | $2 |
| **Total** | **$66** |

### 9.2 Per-Video Cost
- Processing time: 3 minutes average
- Cost per video: $0.15
- Daily cost (15 videos): $2.25

---

## 10. Future Enhancements

### Phase 2 (Optional)
- Support for longer videos (>60 seconds)
- Multiple language subtitle detection
- Batch upload interface
- Processing analytics dashboard

### Phase 3 (Optional)
- 4K video support
- Real-time processing option
- Mobile app integration
- Advanced queue management

---

## 11. Acceptance Criteria

### MVP Completion
- [ ] API endpoints functional
- [ ] GPU worker processing videos
- [ ] Webhook notifications working
- [ ] Local development environment ready
- [ ] Documentation complete
- [ ] 5 successful test videos processed
- [ ] Cost tracking confirmed <$100/month

### Production Ready
- [ ] SSL certificates configured
- [ ] Monitoring alerts set up
- [ ] Backup strategy implemented
- [ ] Team trained on operations
- [ ] 50 videos processed successfully

---

## 12. Appendices

### A. API Client Examples

#### Python Client - Complete Example
```python
# vsr_client.py
import requests
import time
import os

class VSRClient:
    def __init__(self, api_url, api_key):
        self.api_url = api_url
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def process_local_file(self, file_path, mode="STTN", callback_url=None):
        """Process a local video file"""
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'video/mp4')}
            data = {
                'mode': mode,
                'callback_url': callback_url or f"{self.api_url}/webhook"
            }
            
            response = requests.post(
                f"{self.api_url}/api/upload-and-submit",
                files=files,
                data=data,
                headers=self.headers
            )
        
        response.raise_for_status()
        return response.json()
    
    def process_url(self, video_url, mode="STTN", callback_url=None):
        """Process a video from URL"""
        response = requests.post(
            f"{self.api_url}/api/submit-video",
            json={
                "video_url": video_url,
                "mode": mode,
                "callback_url": callback_url or f"{self.api_url}/webhook"
            },
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def check_status(self, job_id):
        """Check job status"""
        response = requests.get(
            f"{self.api_url}/api/job-status/{job_id}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def wait_for_completion(self, job_id, timeout=300):
        """Wait for job to complete with timeout"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.check_status(job_id)
            
            if status['status'] == 'completed':
                return status
            elif status['status'] == 'failed':
                raise Exception(f"Job failed: {status['error_message']}")
            
            time.sleep(5)
        
        raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")

# Usage example
if __name__ == "__main__":
    client = VSRClient("http://localhost:8000", "local-dev-key")
    
    # Process local file
    result = client.process_local_file("/path/to/video.mp4")
    print(f"Job submitted: {result['job_id']}")
    
    # Wait for completion
    final_status = client.wait_for_completion(result['job_id'])
    print(f"Video ready at: {final_status['video_url']}")
```

#### JavaScript/Node.js Example
```javascript
// vsr-client.js
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

class VSRClient {
  constructor(apiUrl, apiKey) {
    this.apiUrl = apiUrl;
    this.headers = { 'Authorization': `Bearer ${apiKey}` };
  }

  async processLocalFile(filePath, mode = 'STTN', callbackUrl = null) {
    const form = new FormData();
    form.append('file', fs.createReadStream(filePath));
    form.append('mode', mode);
    form.append('callback_url', callbackUrl || `${this.apiUrl}/webhook`);

    const response = await axios.post(
      `${this.apiUrl}/api/upload-and-submit`,
      form,
      {
        headers: {
          ...this.headers,
          ...form.getHeaders()
        }
      }
    );

    return response.data;
  }

  async checkStatus(jobId) {
    const response = await axios.get(
      `${this.apiUrl}/api/job-status/${jobId}`,
      { headers: this.headers }
    );
    return response.data;
  }
}

// Usage
const client = new VSRClient('http://localhost:8000', 'local-dev-key');

client.processLocalFile('./video.mp4')
  .then(result => {
    console.log('Job ID:', result.job_id);
    // Poll for status or wait for webhook
  })
  .catch(error => console.error('Error:', error));
```

#### cURL Commands
```bash
# Upload local file
curl -X POST http://localhost:8000/api/upload-and-submit \
  -H "Authorization: Bearer local-dev-key" \
  -F "file=@/path/to/video.mp4" \
  -F "mode=STTN" \
  -F "callback_url=http://localhost:8001/webhook"

# Submit URL
curl -X POST http://localhost:8000/api/submit-video \
  -H "Authorization: Bearer local-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://example.com/video.mp4",
    "mode": "STTN",
    "callback_url": "http://localhost:8001/webhook"
  }'

# Check status
curl -X GET http://localhost:8000/api/job-status/{job_id} \
  -H "Authorization: Bearer local-dev-key"
```

#### Postman Collection
```json
{
  "info": {
    "name": "VSR API",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "auth": {
    "type": "bearer",
    "bearer": [{"key": "token", "value": "{{api_key}}"}]
  },
  "variable": [
    {"key": "base_url", "value": "http://localhost:8000"},
    {"key": "api_key", "value": "local-dev-key"}
  ],
  "item": [
    {
      "name": "Upload and Submit Local File",
      "request": {
        "method": "POST",
        "url": "{{base_url}}/api/upload-and-submit",
        "body": {
          "mode": "formdata",
          "formdata": [
            {"key": "file", "type": "file", "src": "/path/to/video.mp4"},
            {"key": "mode", "value": "STTN"},
            {"key": "callback_url", "value": "http://localhost:8001/webhook"}
          ]
        }
      }
    }
  ]
}
```

### B. Error Codes
| Code | Description |
|------|-------------|
| 400 | Invalid request |
| 401 | Invalid API key |
| 404 | Job not found |
| 413 | Video too large |
| 500 | Server error |

### C. Glossary
- **VSR**: Video Subtitle Remover
- **STTN**: Spatial-Temporal Transformer Network (fast mode)
- **LAMA**: Large Mask Inpainting (quality mode)
- **PROPAINTER**: Advanced propagation model (premium mode)