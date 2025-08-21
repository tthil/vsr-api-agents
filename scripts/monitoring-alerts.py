#!/usr/bin/env python3
"""
Monitoring and alerting system for VSR API.

Implements proactive monitoring with DigitalOcean uptime monitoring,
queue depth alerts, and notification systems via email/Slack.
"""

import asyncio
import aiohttp
import smtplib
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import structlog

logger = structlog.get_logger(__name__)


class AlertManager:
    """
    Manages alerts and notifications for VSR API monitoring.
    
    Supports email and Slack notifications with configurable thresholds
    and alert fatigue prevention.
    """
    
    def __init__(self):
        # Configuration from environment
        self.api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        self.alert_email = os.getenv("ALERT_EMAIL")
        self.slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        
        # SMTP configuration
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        
        # Alert thresholds
        self.queue_depth_warning = int(os.getenv("QUEUE_DEPTH_WARNING", "50"))
        self.queue_depth_critical = int(os.getenv("QUEUE_DEPTH_CRITICAL", "100"))
        self.error_rate_warning = float(os.getenv("ERROR_RATE_WARNING", "5.0"))  # %
        self.error_rate_critical = float(os.getenv("ERROR_RATE_CRITICAL", "10.0"))  # %
        self.response_time_warning = float(os.getenv("RESPONSE_TIME_WARNING", "2.0"))  # seconds
        self.webhook_failure_rate_warning = float(os.getenv("WEBHOOK_FAILURE_RATE_WARNING", "20.0"))  # %
        
        # Alert state tracking (prevent spam)
        self.alert_states = {}
        self.alert_cooldown = 300  # 5 minutes between same alerts
        
    async def check_system_health(self) -> Dict[str, Any]:
        """
        Comprehensive system health check.
        
        Returns:
            Dictionary with health status and metrics
        """
        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_healthy": True,
            "alerts": [],
            "metrics": {}
        }
        
        try:
            # Check API health
            api_health = await self._check_api_health()
            health_status["metrics"]["api"] = api_health
            
            if not api_health["healthy"]:
                health_status["overall_healthy"] = False
                health_status["alerts"].append({
                    "severity": "critical",
                    "type": "api_down",
                    "message": "API health check failed",
                    "details": api_health
                })
            
            # Check metrics and thresholds
            metrics = await self._get_api_metrics()
            health_status["metrics"]["performance"] = metrics
            
            # Queue depth alerts
            queue_depth = metrics.get("queue_depth", 0)
            if queue_depth >= self.queue_depth_critical:
                health_status["alerts"].append({
                    "severity": "critical",
                    "type": "queue_backup_critical",
                    "message": f"Queue depth critically high: {queue_depth} jobs",
                    "details": {"queue_depth": queue_depth, "threshold": self.queue_depth_critical}
                })
            elif queue_depth >= self.queue_depth_warning:
                health_status["alerts"].append({
                    "severity": "warning",
                    "type": "queue_backup_warning",
                    "message": f"Queue depth high: {queue_depth} jobs",
                    "details": {"queue_depth": queue_depth, "threshold": self.queue_depth_warning}
                })
            
            # Error rate alerts
            error_rate = metrics.get("error_rate_percent", 0)
            if error_rate >= self.error_rate_critical:
                health_status["overall_healthy"] = False
                health_status["alerts"].append({
                    "severity": "critical",
                    "type": "high_error_rate",
                    "message": f"Error rate critically high: {error_rate:.1f}%",
                    "details": {"error_rate": error_rate, "threshold": self.error_rate_critical}
                })
            elif error_rate >= self.error_rate_warning:
                health_status["alerts"].append({
                    "severity": "warning",
                    "type": "elevated_error_rate",
                    "message": f"Error rate elevated: {error_rate:.1f}%",
                    "details": {"error_rate": error_rate, "threshold": self.error_rate_warning}
                })
            
            # Response time alerts
            avg_response_time = metrics.get("avg_response_time_seconds", 0)
            if avg_response_time >= self.response_time_warning:
                health_status["alerts"].append({
                    "severity": "warning",
                    "type": "slow_response_time",
                    "message": f"Response time slow: {avg_response_time:.2f}s",
                    "details": {"response_time": avg_response_time, "threshold": self.response_time_warning}
                })
            
            # Webhook failure rate alerts
            webhook_failure_rate = metrics.get("webhook_failure_rate_percent", 0)
            if webhook_failure_rate >= self.webhook_failure_rate_warning:
                health_status["alerts"].append({
                    "severity": "warning",
                    "type": "webhook_failures",
                    "message": f"Webhook failure rate high: {webhook_failure_rate:.1f}%",
                    "details": {"failure_rate": webhook_failure_rate, "threshold": self.webhook_failure_rate_warning}
                })
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            health_status["overall_healthy"] = False
            health_status["alerts"].append({
                "severity": "critical",
                "type": "monitoring_failure",
                "message": f"Health check system failure: {str(e)}",
                "details": {"error": str(e)}
            })
        
        return health_status
    
    async def process_alerts(self, health_status: Dict[str, Any]):
        """Process and send alerts based on health status."""
        for alert in health_status["alerts"]:
            await self._send_alert(alert)
    
    async def _check_api_health(self) -> Dict[str, Any]:
        """Check API health endpoint."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                # Check basic health
                async with session.get(f"{self.api_base_url}/health/healthz") as response:
                    if response.status == 200:
                        health_data = await response.json()
                        
                        # Check readiness
                        async with session.get(f"{self.api_base_url}/health/readyz") as ready_response:
                            ready_data = await ready_response.json() if ready_response.status == 200 else {}
                            
                            return {
                                "healthy": True,
                                "status_code": response.status,
                                "uptime_seconds": health_data.get("uptime_seconds", 0),
                                "services": ready_data.get("services", {}),
                                "response_time_ms": 0  # Would need to measure this
                            }
                    else:
                        return {
                            "healthy": False,
                            "status_code": response.status,
                            "error": f"Health check returned {response.status}"
                        }
                        
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }
    
    async def _get_api_metrics(self) -> Dict[str, Any]:
        """Get API performance metrics."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(f"{self.api_base_url}/metrics/json") as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning("Failed to get metrics", status_code=response.status)
                        return {}
        except Exception as e:
            logger.error("Failed to get API metrics", error=str(e))
            return {}
    
    async def _send_alert(self, alert: Dict[str, Any]):
        """Send alert via configured channels."""
        alert_key = f"{alert['type']}_{alert['severity']}"
        current_time = time.time()
        
        # Check cooldown to prevent spam
        if alert_key in self.alert_states:
            last_sent = self.alert_states[alert_key]
            if current_time - last_sent < self.alert_cooldown:
                return
        
        # Update alert state
        self.alert_states[alert_key] = current_time
        
        # Format alert message
        message = self._format_alert_message(alert)
        
        # Send via email
        if self.alert_email and self.smtp_username and self.smtp_password:
            await self._send_email_alert(alert, message)
        
        # Send via Slack
        if self.slack_webhook_url:
            await self._send_slack_alert(alert, message)
        
        logger.info(
            "Alert sent",
            type=alert["type"],
            severity=alert["severity"],
            message=alert["message"]
        )
    
    def _format_alert_message(self, alert: Dict[str, Any]) -> str:
        """Format alert message for notifications."""
        severity_emoji = {
            "critical": "🚨",
            "warning": "⚠️",
            "info": "ℹ️"
        }
        
        emoji = severity_emoji.get(alert["severity"], "🔔")
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        message = f"{emoji} VSR API Alert - {alert['severity'].upper()}\n\n"
        message += f"Type: {alert['type']}\n"
        message += f"Message: {alert['message']}\n"
        message += f"Time: {timestamp}\n"
        
        if alert.get("details"):
            message += f"\nDetails:\n"
            for key, value in alert["details"].items():
                message += f"  {key}: {value}\n"
        
        return message
    
    async def _send_email_alert(self, alert: Dict[str, Any], message: str):
        """Send alert via email."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_username
            msg['To'] = self.alert_email
            msg['Subject'] = f"VSR API Alert: {alert['type']} ({alert['severity']})"
            
            msg.attach(MIMEText(message, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)
            server.send_message(msg)
            server.quit()
            
            logger.info("Email alert sent", recipient=self.alert_email)
            
        except Exception as e:
            logger.error("Failed to send email alert", error=str(e))
    
    async def _send_slack_alert(self, alert: Dict[str, Any], message: str):
        """Send alert via Slack webhook."""
        try:
            color_map = {
                "critical": "danger",
                "warning": "warning",
                "info": "good"
            }
            
            payload = {
                "text": f"VSR API Alert: {alert['type']}",
                "attachments": [
                    {
                        "color": color_map.get(alert["severity"], "warning"),
                        "fields": [
                            {
                                "title": "Severity",
                                "value": alert["severity"].upper(),
                                "short": True
                            },
                            {
                                "title": "Type",
                                "value": alert["type"],
                                "short": True
                            },
                            {
                                "title": "Message",
                                "value": alert["message"],
                                "short": False
                            }
                        ],
                        "ts": int(time.time())
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.slack_webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.info("Slack alert sent")
                    else:
                        logger.error("Failed to send Slack alert", status_code=response.status)
                        
        except Exception as e:
            logger.error("Failed to send Slack alert", error=str(e))


async def run_monitoring_cycle():
    """Run a single monitoring cycle."""
    alert_manager = AlertManager()
    
    logger.info("Starting monitoring cycle")
    
    # Check system health
    health_status = await alert_manager.check_system_health()
    
    # Log health status
    logger.info(
        "Health check completed",
        healthy=health_status["overall_healthy"],
        alerts_count=len(health_status["alerts"])
    )
    
    # Process alerts
    if health_status["alerts"]:
        await alert_manager.process_alerts(health_status)
    
    return health_status


async def main():
    """Main monitoring loop."""
    logger.info("VSR API Monitoring started")
    
    # Monitoring interval (5 minutes default)
    interval = int(os.getenv("MONITORING_INTERVAL", "300"))
    
    while True:
        try:
            await run_monitoring_cycle()
            await asyncio.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
            break
        except Exception as e:
            logger.error("Monitoring cycle failed", error=str(e))
            await asyncio.sleep(60)  # Wait 1 minute before retry


if __name__ == "__main__":
    asyncio.run(main())
