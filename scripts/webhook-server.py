#!/usr/bin/env python3
"""Local webhook test server for VSR API development."""

import hashlib
import hmac
import json
import time
from datetime import datetime
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import JSONResponse

app = FastAPI(title="VSR Webhook Test Server", version="1.0.0")

# Configuration
WEBHOOK_SECRET = "default-webhook-secret-change-in-production"
received_webhooks = []  # Store received webhooks for inspection


def verify_webhook_signature(payload: bytes, signature: str, timestamp: str) -> bool:
    """Verify webhook signature matches expected HMAC-SHA256.
    
    Args:
        payload: Raw request body bytes
        signature: Signature from X-VSR-Signature header (format: sha256=<hex>)
        timestamp: Timestamp from X-VSR-Timestamp header
        
    Returns:
        True if signature is valid, False otherwise
    """
    if not signature.startswith("sha256="):
        return False
    
    received_signature = signature[7:]  # Remove "sha256=" prefix
    
    # Create signature string: timestamp + payload
    sig_string = f"{timestamp}.{payload.decode('utf-8')}"
    
    # Generate expected HMAC-SHA256
    expected_signature = hmac.new(
        WEBHOOK_SECRET.encode('utf-8'),
        sig_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    # Use constant-time comparison to prevent timing attacks
    return hmac.compare_digest(received_signature, expected_signature)


@app.post("/webhook")
async def receive_webhook(
    request: Request,
    x_vsr_signature: str = Header(..., alias="X-VSR-Signature"),
    x_vsr_timestamp: str = Header(..., alias="X-VSR-Timestamp"),
):
    """Receive and validate VSR webhook notifications."""
    
    # Read raw body for signature verification
    body = await request.body()
    
    # Verify timestamp is recent (within 5 minutes)
    try:
        webhook_time = int(x_vsr_timestamp)
        current_time = int(time.time())
        if abs(current_time - webhook_time) > 300:  # 5 minutes
            raise HTTPException(
                status_code=400,
                detail="Webhook timestamp too old or in future"
            )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid timestamp format")
    
    # Verify signature
    if not verify_webhook_signature(body, x_vsr_signature, x_vsr_timestamp):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")
    
    # Parse JSON payload
    try:
        payload = json.loads(body.decode('utf-8'))
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    
    # Store webhook for inspection
    webhook_record = {
        "received_at": datetime.utcnow().isoformat(),
        "signature": x_vsr_signature,
        "timestamp": x_vsr_timestamp,
        "payload": payload,
        "headers": dict(request.headers),
    }
    received_webhooks.append(webhook_record)
    
    # Keep only last 100 webhooks
    if len(received_webhooks) > 100:
        received_webhooks.pop(0)
    
    # Log webhook receipt
    print(f"[{datetime.utcnow().isoformat()}] Received webhook:")
    print(f"  Job ID: {payload.get('job_id')}")
    print(f"  Status: {payload.get('status')}")
    print(f"  Video Key: {payload.get('video_key')}")
    if payload.get('processed_video_key'):
        print(f"  Processed Video Key: {payload.get('processed_video_key')}")
    if payload.get('error_message'):
        print(f"  Error: {payload.get('error_message')}")
    if payload.get('processing_time_seconds'):
        print(f"  Processing Time: {payload.get('processing_time_seconds')}s")
    print()
    
    return JSONResponse(
        status_code=200,
        content={"status": "received", "job_id": payload.get("job_id")}
    )


@app.get("/webhooks")
async def list_received_webhooks():
    """List all received webhooks for inspection."""
    return {
        "count": len(received_webhooks),
        "webhooks": received_webhooks
    }


@app.get("/webhooks/{job_id}")
async def get_webhook_by_job_id(job_id: str):
    """Get webhook by job ID."""
    matching_webhooks = [
        w for w in received_webhooks 
        if w["payload"].get("job_id") == job_id
    ]
    
    if not matching_webhooks:
        raise HTTPException(status_code=404, detail="No webhook found for job ID")
    
    return {
        "job_id": job_id,
        "webhooks": matching_webhooks
    }


@app.delete("/webhooks")
async def clear_webhooks():
    """Clear all received webhooks."""
    global received_webhooks
    count = len(received_webhooks)
    received_webhooks.clear()
    return {"message": f"Cleared {count} webhooks"}


@app.get("/")
async def root():
    """Root endpoint with server information."""
    return {
        "name": "VSR Webhook Test Server",
        "version": "1.0.0",
        "webhook_endpoint": "/webhook",
        "received_count": len(received_webhooks),
        "webhook_secret_configured": bool(WEBHOOK_SECRET),
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


if __name__ == "__main__":
    print("🚀 Starting VSR Webhook Test Server")
    print(f"📡 Webhook endpoint: http://localhost:8001/webhook")
    print(f"🔍 Inspection endpoint: http://localhost:8001/webhooks")
    print(f"🔐 Using webhook secret: {WEBHOOK_SECRET[:10]}...")
    print()
    
    uvicorn.run(
        "webhook-server:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
