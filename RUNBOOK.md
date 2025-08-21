# 🚨 VSR API Operations Runbook

> **Comprehensive troubleshooting guide and operational procedures for the Video Subtitle Removal API**

## 📋 Quick Reference

### **Emergency Contacts**
- **On-Call Engineer**: [Your team contact]
- **DevOps Team**: [DevOps contact]
- **DigitalOcean Support**: [Support ticket system]

### **Critical Services**
- **API**: `https://your-domain.com/api`
- **Health Check**: `https://your-domain.com/health/healthz`
- **Metrics**: `https://your-domain.com/metrics`
- **Monitoring Dashboard**: [Grafana/monitoring URL]

---

## 🔥 **Critical Issues (P0)**

### **Issue: API Completely Down**

**Symptoms:**
- Health check returns 5xx or times out
- All API endpoints unresponsive
- High error rate in monitoring

**Immediate Actions:**
1. **Check service status**:
   ```bash
   curl -I https://your-domain.com/health/healthz
   # Expected: HTTP/1.1 200 OK
   ```

2. **Check Docker containers**:
   ```bash
   docker-compose -f infra/docker-compose.prod.yml ps
   # All services should show "Up (healthy)"
   ```

3. **Restart API service**:
   ```bash
   docker-compose -f infra/docker-compose.prod.yml restart api
   ```

4. **Check logs**:
   ```bash
   docker-compose -f infra/docker-compose.prod.yml logs api --tail=50
   ```

**Root Cause Investigation:**
- Database connectivity issues
- Memory/CPU exhaustion
- SSL certificate expiration
- Load balancer misconfiguration

---

### **Issue: GPU Worker Out of Memory (OOM)**

**Symptoms:**
- Jobs stuck in "processing" status
- Worker container restarts frequently
- CUDA out of memory errors in logs

**Immediate Actions:**
1. **Check worker logs**:
   ```bash
   docker-compose logs worker --tail=100 | grep -i "memory\|cuda\|oom"
   ```

2. **Check GPU memory usage**:
   ```bash
   nvidia-smi
   # Look for high memory usage or zombie processes
   ```

3. **Restart worker with memory cleanup**:
   ```bash
   docker-compose restart worker
   # This clears GPU memory
   ```

4. **Reduce batch size** (temporary fix):
   ```bash
   # Edit worker environment variables
   BATCH_SIZE=1  # Reduce from default
   PROCESSING_THREADS=1  # Reduce parallelism
   ```

**Prevention:**
- Monitor GPU memory usage
- Implement batch size auto-adjustment
- Set memory limits in Docker Compose

---

### **Issue: Queue Backup (High Job Volume)**

**Symptoms:**
- Queue depth > 100 jobs
- ETA estimates > 2 hours
- Customer complaints about processing delays

**Immediate Actions:**
1. **Check queue depth**:
   ```bash
   curl -s https://your-domain.com/metrics/json | jq '.queue_depth'
   ```

2. **Scale worker horizontally**:
   ```bash
   docker-compose -f infra/docker-compose.prod.yml up -d --scale worker=3
   ```

3. **Enable business hours bypass** (if needed):
   ```bash
   # Temporarily disable business hours restriction
   export BUSINESS_HOURS_ONLY=false
   docker-compose restart api
   ```

4. **Monitor processing rate**:
   ```bash
   # Check jobs completed per hour
   curl -s https://your-domain.com/metrics/json | jq '.jobs_completed_hourly'
   ```

---

## ⚠️ **High Priority Issues (P1)**

### **Issue: Webhook Delivery Failures**

**Symptoms:**
- High webhook failure rate in metrics
- Customer reports of missing notifications
- Webhook retry exhaustion in logs

**Diagnosis:**
1. **Check webhook metrics**:
   ```bash
   curl -s https://your-domain.com/metrics/json | jq '.webhook_failures'
   ```

2. **Review webhook logs**:
   ```bash
   docker-compose logs worker | grep -i webhook
   ```

3. **Test webhook connectivity**:
   ```bash
   # Test from worker container
   docker exec worker curl -I https://customer-webhook-url.com/webhook
   ```

**Resolution:**
1. **Verify customer webhook endpoints**
2. **Check network connectivity and DNS**
3. **Review webhook signature validation**
4. **Increase retry attempts if needed**

---

### **Issue: Database Connection Issues**

**Symptoms:**
- Intermittent 5xx errors
- "Connection pool exhausted" errors
- Slow API response times

**Diagnosis:**
1. **Check database connectivity**:
   ```bash
   curl -s https://your-domain.com/health/readyz | jq '.services.mongodb'
   ```

2. **Check MongoDB logs**:
   ```bash
   docker-compose logs mongodb --tail=50
   ```

3. **Monitor connection pool**:
   ```bash
   # Check active connections
   docker exec mongodb mongo --eval "db.serverStatus().connections"
   ```

**Resolution:**
1. **Restart MongoDB if needed**:
   ```bash
   docker-compose restart mongodb
   ```

2. **Increase connection pool size**:
   ```bash
   # Edit MongoDB configuration
   MONGODB_MAX_POOL_SIZE=50  # Increase from default
   ```

3. **Check for long-running queries**:
   ```bash
   docker exec mongodb mongo --eval "db.currentOp()"
   ```

---

## 🔧 **Medium Priority Issues (P2)**

### **Issue: High API Error Rate**

**Symptoms:**
- 4xx/5xx error rate > 5%
- Customer complaints about failed uploads
- Quota exceeded errors

**Diagnosis:**
1. **Check error breakdown**:
   ```bash
   curl -s https://your-domain.com/metrics/json | jq '.errors'
   ```

2. **Review API logs**:
   ```bash
   docker-compose logs api | grep -E "4[0-9]{2}|5[0-9]{2}"
   ```

**Common Causes & Solutions:**
- **413 Request Too Large**: Increase `MAX_REQUEST_SIZE`
- **429 Rate Limited**: Review quota settings
- **422 Validation Error**: Check video format validation
- **503 Service Unavailable**: Check service health

---

### **Issue: Storage Space Issues**

**Symptoms:**
- Upload failures
- Processing errors
- Disk space alerts

**Diagnosis:**
1. **Check Spaces usage**:
   ```bash
   # Check bucket size and object count
   mc du spaces/vsr-videos
   ```

2. **Check local disk usage**:
   ```bash
   df -h
   docker system df
   ```

**Resolution:**
1. **Run cleanup script**:
   ```bash
   python scripts/cleanup-storage.py --dry-run
   python scripts/cleanup-storage.py --execute
   ```

2. **Clean Docker resources**:
   ```bash
   docker system prune -f
   docker volume prune -f
   ```

---

## 📊 **Monitoring & Alerting**

### **Key Metrics to Monitor**

1. **API Health**:
   - Response time < 2s (95th percentile)
   - Error rate < 1%
   - Uptime > 99.9%

2. **Queue Performance**:
   - Queue depth < 50 jobs
   - Processing time < 30s per job
   - Worker utilization 70-90%

3. **Resource Usage**:
   - CPU usage < 80%
   - Memory usage < 85%
   - GPU memory < 90%
   - Disk usage < 80%

### **Alert Thresholds**

```yaml
# Example alert configuration
alerts:
  critical:
    - api_down: "Health check fails for > 2 minutes"
    - high_error_rate: "Error rate > 10% for > 5 minutes"
    - gpu_oom: "GPU memory > 95% for > 1 minute"
    
  warning:
    - queue_backup: "Queue depth > 100 jobs"
    - slow_response: "95th percentile > 5s for > 10 minutes"
    - webhook_failures: "Webhook failure rate > 20%"
```

---

## 💾 **Backup & Recovery**

### **Daily Backup Procedure**

1. **MongoDB Backup**:
   ```bash
   # Automated daily backup script
   #!/bin/bash
   DATE=$(date +%Y%m%d)
   docker exec mongodb mongodump --out /backup/mongodb-$DATE
   mc cp -r /backup/mongodb-$DATE spaces/vsr-backups/mongodb/
   ```

2. **Configuration Backup**:
   ```bash
   # Backup configuration files
   tar -czf config-backup-$DATE.tar.gz \
     infra/ \
     .env.prod \
     docker-compose.prod.yml
   mc cp config-backup-$DATE.tar.gz spaces/vsr-backups/config/
   ```

### **Recovery Procedures**

1. **MongoDB Recovery**:
   ```bash
   # Stop services
   docker-compose down
   
   # Restore from backup
   mc cp spaces/vsr-backups/mongodb/mongodb-YYYYMMDD.tar.gz ./
   tar -xzf mongodb-YYYYMMDD.tar.gz
   docker-compose up -d mongodb
   docker exec mongodb mongorestore /backup/mongodb-YYYYMMDD
   
   # Restart all services
   docker-compose up -d
   ```

2. **Full System Recovery**:
   ```bash
   # Clone repository
   git clone https://github.com/your-org/vsr-api.git
   cd vsr-api
   
   # Restore configuration
   mc cp spaces/vsr-backups/config/config-backup-YYYYMMDD.tar.gz ./
   tar -xzf config-backup-YYYYMMDD.tar.gz
   
   # Deploy
   docker-compose -f infra/docker-compose.prod.yml up -d
   ```

---

## 🔍 **Debugging Commands**

### **Service Health Checks**
```bash
# Quick health overview
curl -s https://your-domain.com/health/readyz | jq

# Detailed metrics
curl -s https://your-domain.com/metrics/json | jq

# Container status
docker-compose ps

# Resource usage
docker stats --no-stream
```

### **Log Analysis**
```bash
# API errors
docker-compose logs api | grep ERROR | tail -20

# Worker processing
docker-compose logs worker | grep "Processing job" | tail -10

# Database queries
docker-compose logs mongodb | grep "slow query"

# Webhook deliveries
docker-compose logs worker | grep webhook | tail -20
```

### **Performance Analysis**
```bash
# API response times
curl -w "@curl-format.txt" -s -o /dev/null https://your-domain.com/api/health

# Database performance
docker exec mongodb mongo --eval "db.jobs.explain('executionStats').find()"

# Queue depth over time
watch -n 5 'curl -s https://your-domain.com/metrics/json | jq .queue_depth'
```

---

## 🚀 **Deployment Procedures**

### **Rolling Deployment**
```bash
# 1. Pull latest code
git pull origin main

# 2. Build new images
docker-compose -f infra/docker-compose.prod.yml build

# 3. Update API (zero downtime)
docker-compose -f infra/docker-compose.prod.yml up -d --no-deps api

# 4. Update worker (drain existing jobs first)
docker-compose -f infra/docker-compose.prod.yml stop worker
# Wait for jobs to complete
docker-compose -f infra/docker-compose.prod.yml up -d worker

# 5. Verify deployment
curl https://your-domain.com/health/healthz
```

### **Rollback Procedure**
```bash
# 1. Identify last known good version
git log --oneline -10

# 2. Checkout previous version
git checkout <previous-commit-hash>

# 3. Rebuild and deploy
docker-compose -f infra/docker-compose.prod.yml build
docker-compose -f infra/docker-compose.prod.yml up -d

# 4. Verify rollback
curl https://your-domain.com/health/healthz
```

---

## 📞 **Escalation Procedures**

### **Severity Levels**

**P0 - Critical (Immediate Response)**
- Complete service outage
- Data loss or corruption
- Security breach
- Response time: < 15 minutes

**P1 - High (1 Hour Response)**
- Partial service degradation
- High error rates
- Performance issues affecting customers
- Response time: < 1 hour

**P2 - Medium (4 Hour Response)**
- Minor functionality issues
- Non-critical feature failures
- Response time: < 4 hours

**P3 - Low (Next Business Day)**
- Enhancement requests
- Documentation updates
- Response time: Next business day

### **Contact Information**
```
Primary On-Call: [Phone] [Email] [Slack]
Secondary On-Call: [Phone] [Email] [Slack]
Engineering Manager: [Phone] [Email] [Slack]
DevOps Team: [Email] [Slack Channel]
```

---

## 📚 **Additional Resources**

- **Architecture Documentation**: `docs/ARCHITECTURE.md`
- **API Documentation**: `https://your-domain.com/docs`
- **Monitoring Dashboard**: [Grafana URL]
- **Log Aggregation**: [Logging system URL]
- **Incident Management**: [Incident tracking system]
- **Team Wiki**: [Internal documentation]

---

**Last Updated**: 2025-08-13  
**Version**: 1.0  
**Maintained By**: VSR Engineering Team
