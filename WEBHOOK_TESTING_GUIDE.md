# 🔗 VSR API Webhook Testing Guide

> **Complete step-by-step guide for testing the webhook delivery subsystem**

## 📋 Overview

The VSR API includes a robust webhook delivery subsystem that sends HTTP notifications when video processing jobs complete or fail. This guide provides comprehensive instructions for testing the webhook functionality in your local development environment.

## 🎯 What You'll Test

- **Webhook Delivery**: HTTP POST notifications sent on job completion
- **HMAC Security**: SHA256 signature verification for payload authenticity
- **Retry Logic**: Exponential backoff for failed deliveries
- **Error Handling**: Proper logging and database event tracking
- **Local Development**: HTTP callback URLs for easy testing

---

## 📋 Prerequisites

Before starting, ensure you have:

- ✅ **Docker & Docker Compose** installed
- ✅ **VSR API project** cloned locally
- ✅ **Test video file** at `vids/source.mp4`
- ✅ **Python 3.11+** for webhook server
- ✅ **curl** and **jq** for API testing

---

## 🚀 Step-by-Step Testing Instructions

### **Step 1: Start All VSR Services**

```bash
# Navigate to project root
cd /path/to/vsr-api

# Start all services (API, worker, databases)
docker-compose -f infra/docker-compose.local.yml up -d

# Verify all services are healthy (wait ~30 seconds)
docker-compose -f infra/docker-compose.local.yml ps
```

**✅ Expected Output:**
```
NAME               IMAGE              STATUS
vsr-api           vsr-api:latest     Up (healthy)
vsr-mock-worker   vsr-worker:latest  Up (healthy)  
vsr-mongodb       mongo:7.0          Up (healthy)
vsr-rabbitmq      rabbitmq:3.12      Up (healthy)
vsr-minio         minio/minio        Up (healthy)
```

**❌ If services fail to start:**
- Check Docker daemon is running
- Ensure ports 8000, 5432, 5672, 9000 are available
- Run `docker-compose logs [service-name]` to debug

### **Step 2: Start the Webhook Test Server**

**Open a NEW terminal window** and start the webhook receiver:

```bash
# Navigate to project root
cd /path/to/vsr-api

# Start webhook test server
python scripts/webhook-server.py
```

**✅ Expected Output:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
```

**⚠️ IMPORTANT:** Keep this terminal open! The webhook server must be running to receive notifications.

**❌ If webhook server fails:**
```bash
# Install missing dependencies
pip install fastapi uvicorn

# Or use different port if 8001 is busy
python scripts/webhook-server.py --port 8002
```

### **Step 3: Submit Job with Webhook Callback**

In your **original terminal**, submit a video processing job:

```bash
# Submit job with webhook callback
curl -X POST "http://localhost:8000/api/upload-and-submit" \
  -F "video_file=@vids/source.mp4;type=video/mp4" \
  -F "mode=sttn" \
  -F "callback_url=http://host.docker.internal:8001/webhook"
```

**✅ Expected Response:**
```json
{
  "id": "c8090ae9-43d5-4088-b9ec-742bc99ccd0d",
  "status": "pending",
  "mode": "sttn",
  "subtitle_area": null,
  "video_key": "KeyPrefix.UPLOADS/20250813/c8090ae9-43d5-4088-b9ec-742bc99ccd0d/video.mp4",
  "queue_position": 2,
  "eta_seconds": 150,
  "created_at": "2025-08-13T09:38:07.704754",
  "callback_url": "http://host.docker.internal:8001/webhook"
}
```

**📝 SAVE THE JOB ID** - you'll need it for monitoring!

**🔑 Key Points:**
- Use `host.docker.internal:8001` (not `localhost`) for Docker networking
- The callback URL must be reachable from the worker container
- Job processing takes ~5-10 seconds in mock mode

### **Step 4: Monitor Job Processing**

Watch the worker process your job in real-time:

```bash
# Monitor worker logs
docker-compose -f infra/docker-compose.local.yml logs mock-worker --follow
```

**✅ Expected Log Progression:**
```
vsr-mock-worker  | Processing job: c8090ae9-43d5-4088-b9ec-742bc99ccd0d
vsr-mock-worker  | Job c8090ae9-43d5-4088-b9ec-742bc99ccd0d: Starting video processing...
vsr-mock-worker  | Webhook notifier initialized
vsr-mock-worker  | Sending webhook job_id=c8090ae9-43d5-4088-b9ec-742bc99ccd0d status=completed url=http://host.docker.internal:8001/webhook
vsr-mock-worker  | HTTP Request: POST http://host.docker.internal:8001/webhook "HTTP/1.1 200 OK"
vsr-mock-worker  | Webhook delivered successfully delivery_time_ms=31.99 job_id=c8090ae9-43d5-4088-b9ec-742bc99ccd0d status_code=200
vsr-mock-worker  | Webhook event logged event_type=webhook_sent job_id=c8090ae9-43d5-4088-b9ec-742bc99ccd0d success=True
vsr-mock-worker  | Job completion webhook sent job_id=c8090ae9-43d5-4088-b9ec-742bc99ccd0d status_code=200 success=True
vsr-mock-worker  | Job c8090ae9-43d5-4088-b9ec-742bc99ccd0d: Processing completed successfully
```

**🎯 What to Look For:**
- ✅ `Webhook delivered successfully` with HTTP 200
- ✅ `delivery_time_ms` showing response time (~30ms)
- ✅ `Webhook event logged` confirming database persistence
- ✅ No error messages or retry attempts

### **Step 5: Verify Webhook Delivery**

Check if the webhook was received by your test server:

```bash
# Check received webhooks
curl -s "http://localhost:8001/webhooks" | jq
```

**✅ Expected Response:**
```json
{
  "count": 1,
  "webhooks": [
    {
      "received_at": "2025-08-13T07:39:24.916566",
      "signature": "sha256=36417041208221945083dd6ea26199089e15c3600d03064fe7216033449603d5",
      "timestamp": "1755070764",
      "payload": {
        "job_id": "c8090ae9-43d5-4088-b9ec-742bc99ccd0d",
        "status": "completed",
        "timestamp": "2025-08-13T07:39:24.885208",
        "video_key": "KeyPrefix.UPLOADS/20250813/c8090ae9-43d5-4088-b9ec-742bc99ccd0d/video.mp4",
        "processed_video_key": "processed/c8090ae9-43d5-4088-b9ec-742bc99ccd0d.mp4",
        "processing_time_seconds": 5.001475,
        "error_message": null
      },
      "headers": {
        "x-vsr-signature": "sha256=36417041208221945083dd6ea26199089e15c3600d03064fe7216033449603d5",
        "x-vsr-timestamp": "1755070764",
        "content-type": "application/json",
        "user-agent": "VSR-Webhook/1.0"
      }
    }
  ]
}
```

**🔐 Security Verification:**
- ✅ `x-vsr-signature` header with HMAC-SHA256 signature
- ✅ `x-vsr-timestamp` header for replay attack prevention
- ✅ `user-agent: VSR-Webhook/1.0` identifying the source

### **Step 6: Verify Final Job Status**

Confirm the job completed successfully:

```bash
# Check job status (replace with your actual job ID)
curl -s "http://localhost:8000/api/job-status/1d5ea234-6eb1-4200-af94-db6fb4e756f2" | jq
```

**✅ Expected Response:**
```json
{
  "id": "c8090ae9-43d5-4088-b9ec-742bc99ccd0d",
  "status": "completed",
  "mode": "sttn",
  "video_key": "KeyPrefix.UPLOADS/20250813/c8090ae9-43d5-4088-b9ec-742bc99ccd0d/video.mp4",
  "processed_video_key": "processed/c8090ae9-43d5-4088-b9ec-742bc99ccd0d.mp4",
  "processing_time_seconds": 5.001475,
  "callback_url": "http://host.docker.internal:8001/webhook"
}
```

---

## 🔍 Troubleshooting Common Issues

### **❌ Issue 1: No Webhook Received**

**Symptoms:** `{"count": 0, "webhooks": []}`

**Root Causes & Solutions:**

1. **Webhook server not running**
   ```bash
   # Check if webhook server is active
   curl http://localhost:8001/health
   # Should return: {"status": "healthy"}
   ```

2. **Wrong callback URL**
   ```bash
   # ❌ Wrong: localhost (worker can't reach host)
   "callback_url": "http://localhost:8001/webhook"
   
   # ✅ Correct: Docker networking
   "callback_url": "http://host.docker.internal:8001/webhook"
   ```

3. **Network connectivity issues**
   ```bash
   # Test connectivity from worker container
   docker exec vsr-mock-worker curl -I http://host.docker.internal:8001/health
   ```

4. **Worker container errors**
   ```bash
   # Check worker logs for webhook delivery errors
   docker-compose -f infra/docker-compose.local.yml logs mock-worker --tail=20
   ```

### **❌ Issue 2: Webhook Server Won't Start**

**Symptoms:** `ModuleNotFoundError` or port binding errors

**Solutions:**

1. **Install missing dependencies**
   ```bash
   pip install fastapi uvicorn
   ```

2. **Port already in use**
   ```bash
   # Check what's using port 8001
   lsof -i :8001
   
   # Use different port
   python scripts/webhook-server.py --port 8002
   # Update callback URL accordingly
   ```

3. **Python version issues**
   ```bash
   # Ensure Python 3.11+
   python --version
   ```

### **❌ Issue 3: Worker Not Processing Jobs**

**Symptoms:** Jobs stuck in "pending" status

**Solutions:**

1. **Rebuild worker container**
   ```bash
   docker-compose -f infra/docker-compose.local.yml build mock-worker
   docker-compose -f infra/docker-compose.local.yml up -d mock-worker
   ```

2. **Check RabbitMQ connectivity**
   ```bash
   docker-compose -f infra/docker-compose.local.yml logs rabbitmq
   ```

3. **Verify MongoDB connection**
   ```bash
   docker-compose -f infra/docker-compose.local.yml logs mongodb
   ```

### **❌ Issue 4: Webhook Signature Verification Fails**

**Symptoms:** Invalid signature errors in webhook server logs

**Solutions:**

1. **Check webhook secret consistency**
   ```bash
   # Default secret is "your-webhook-secret-key"
   # Ensure worker and webhook server use same secret
   ```

2. **Verify timestamp tolerance**
   ```bash
   # Check system clock synchronization
   date
   ```

---

## 🧪 Advanced Testing Scenarios

### **Test Multiple Jobs**

```bash
# Submit multiple jobs to test webhook scaling
for i in {1..3}; do
  echo "Submitting job $i..."
  curl -X POST "http://localhost:8000/api/upload-and-submit" \
    -F "video_file=@vids/source.mp4;type=video/mp4" \
    -F "mode=sttn" \
    -F "callback_url=http://host.docker.internal:8001/webhook"
  sleep 2
done

# Check all webhooks received
curl -s "http://localhost:8001/webhooks" | jq '.count'
# Should return: 3
```

### **Test Webhook Failure & Retry Logic**

```bash
# Stop webhook server to simulate failure
# (Ctrl+C in webhook server terminal)

# Submit job - webhook delivery will fail and retry
curl -X POST "http://localhost:8000/api/upload-and-submit" \
  -F "video_file=@vids/source.mp4;type=video/mp4" \
  -F "mode=sttn" \
  -F "callback_url=http://host.docker.internal:8001/webhook"

# Watch worker logs for retry attempts
docker-compose -f infra/docker-compose.local.yml logs mock-worker --follow

# Expected: Multiple retry attempts with exponential backoff
# Then restart webhook server and see successful delivery
```

### **Test Invalid Callback URLs**

```bash
# Test with invalid hostname
curl -X POST "http://localhost:8000/api/upload-and-submit" \
  -F "video_file=@vids/source.mp4;type=video/mp4" \
  -F "mode=sttn" \
  -F "callback_url=http://invalid-host:8001/webhook"

# Check worker logs for proper error handling
docker-compose -f infra/docker-compose.local.yml logs mock-worker --tail=10
```

### **Test Webhook Signature Verification**

```bash
# Get webhook payload and signature
WEBHOOK=$(curl -s "http://localhost:8001/webhooks" | jq -r '.webhooks[0]')
SIGNATURE=$(echo $WEBHOOK | jq -r '.signature')
PAYLOAD=$(echo $WEBHOOK | jq -r '.payload')

echo "Signature: $SIGNATURE"
echo "Payload: $PAYLOAD"

# Manual verification using the webhook secret
# (Implementation depends on your verification needs)
```

---

## ✅ Success Criteria Checklist

Your webhook system is working correctly when you can verify:

- [ ] **Services Started**: All Docker services healthy
- [ ] **Webhook Server Running**: Responds to health checks
- [ ] **Job Submitted**: Returns job ID with callback URL
- [ ] **Job Processed**: Worker logs show processing completion
- [ ] **Webhook Sent**: Worker logs show successful HTTP POST
- [ ] **Webhook Received**: Test server shows webhook in `/webhooks`
- [ ] **Signature Valid**: HMAC-SHA256 signature present and valid
- [ ] **Job Completed**: Final job status is "completed"
- [ ] **Database Logged**: Worker logs show "webhook_sent" event

## 📊 Performance Benchmarks

**Expected Performance Metrics:**

- **Job Processing Time**: 5-10 seconds (mock mode)
- **Webhook Delivery Time**: 20-50ms average
- **HTTP Response**: 200 OK status
- **Retry Attempts**: 0 (on successful delivery)
- **Database Logging**: < 100ms additional overhead

---

## 🔧 Webhook Server API Reference

The test webhook server provides these endpoints:

### **GET /health**
Health check endpoint
```bash
curl http://localhost:8001/health
# Returns: {"status": "healthy"}
```

### **POST /webhook**
Webhook receiver endpoint (used by VSR worker)
```bash
# Automatically called by worker - don't call manually
```

### **GET /webhooks**
List all received webhooks
```bash
curl http://localhost:8001/webhooks
# Returns: {"count": N, "webhooks": [...]}
```

### **DELETE /webhooks**
Clear all received webhooks
```bash
curl -X DELETE http://localhost:8001/webhooks
# Returns: {"message": "All webhooks cleared"}
```

---

## 🎯 Next Steps

After successful webhook testing:

1. **Production Deployment**: Update callback URLs to use HTTPS
2. **Monitoring**: Set up webhook delivery metrics and alerts
3. **Integration**: Integrate webhooks into your application workflow
4. **Security**: Implement proper signature verification in your webhook handlers

---

## 🆘 Getting Help

If you encounter issues not covered in this guide:

1. **Check Docker logs**: `docker-compose logs [service-name]`
2. **Verify network connectivity**: Test with `curl` and `ping`
3. **Review webhook server logs**: Look for error messages
4. **Consult the team**: Share specific error messages and logs

---

## 📚 Additional Resources

- **VSR API Documentation**: `/docs` endpoint when API is running
- **Webhook Implementation**: `shared/vsr_shared/webhooks.py`
- **Worker Integration**: `worker/vsr_worker/queue/simple_consumer.py`
- **Test Server Code**: `scripts/webhook-server.py`

---

**🎉 Happy Testing!** The webhook delivery subsystem is production-ready and thoroughly tested.
