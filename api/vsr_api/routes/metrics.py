"""
Prometheus metrics endpoint for VSR API.

Exposes metrics in Prometheus format for monitoring and alerting.
"""

import time
from typing import Dict, Any
from fastapi import APIRouter, Response
from vsr_api.middleware.metrics import get_metrics
from vsr_shared.db import get_db
from vsr_shared.queue import RabbitMQClient
import structlog

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/metrics", tags=["metrics"])


class PrometheusMetrics:
    """
    Prometheus metrics formatter for VSR API.
    
    Converts internal metrics to Prometheus exposition format.
    """
    
    @staticmethod
    def format_metrics(metrics: Dict[str, Any]) -> str:
        """
        Format metrics in Prometheus exposition format.
        
        Args:
            metrics: Internal metrics dictionary
            
        Returns:
            Prometheus formatted metrics string
        """
        lines = []
        
        # Add metadata
        lines.append("# HELP vsr_info VSR API information")
        lines.append("# TYPE vsr_info gauge")
        lines.append(f'vsr_info{{version="1.0.0"}} 1')
        lines.append("")
        
        # Uptime
        lines.append("# HELP vsr_uptime_seconds VSR API uptime in seconds")
        lines.append("# TYPE vsr_uptime_seconds counter")
        lines.append(f"vsr_uptime_seconds {metrics['uptime_seconds']}")
        lines.append("")
        
        # Request metrics
        lines.append("# HELP vsr_http_requests_total Total HTTP requests")
        lines.append("# TYPE vsr_http_requests_total counter")
        
        for endpoint, count in metrics['requests']['by_endpoint'].items():
            if endpoint != 'total':
                method, path = endpoint.split(':', 1) if ':' in endpoint else ('unknown', endpoint)
                lines.append(f'vsr_http_requests_total{{method="{method}",path="{path}"}} {count}')
        
        lines.append(f"vsr_http_requests_total {metrics['requests']['total']}")
        lines.append("")
        
        # Active requests
        lines.append("# HELP vsr_http_requests_active Currently active HTTP requests")
        lines.append("# TYPE vsr_http_requests_active gauge")
        lines.append(f"vsr_http_requests_active {metrics['requests']['active']}")
        lines.append("")
        
        # Response time metrics
        lines.append("# HELP vsr_http_request_duration_ms HTTP request duration in milliseconds")
        lines.append("# TYPE vsr_http_request_duration_ms histogram")
        
        for endpoint, duration_data in metrics['response_times'].items():
            if endpoint != 'all':
                method, path = endpoint.split(':', 1) if ':' in endpoint else ('unknown', endpoint)
                avg_ms = duration_data['avg_ms']
                lines.append(f'vsr_http_request_duration_ms{{method="{method}",path="{path}",quantile="0.5"}} {avg_ms}')
                lines.append(f'vsr_http_request_duration_ms{{method="{method}",path="{path}",quantile="0.95"}} {duration_data["max_ms"]}')
                lines.append(f'vsr_http_request_duration_ms_count{{method="{method}",path="{path}"}} {duration_data["count"]}')
        
        lines.append("")
        
        # Status code metrics
        lines.append("# HELP vsr_http_responses_total HTTP responses by status code")
        lines.append("# TYPE vsr_http_responses_total counter")
        
        for status_code, count in metrics['status_codes'].items():
            lines.append(f'vsr_http_responses_total{{status_code="{status_code}"}} {count}')
        
        lines.append("")
        
        # Error metrics
        lines.append("# HELP vsr_http_errors_total HTTP errors by type")
        lines.append("# TYPE vsr_http_errors_total counter")
        lines.append(f"vsr_http_errors_total{{type=\"4xx\"}} {metrics['errors']['4xx']}")
        lines.append(f"vsr_http_errors_total{{type=\"5xx\"}} {metrics['errors']['5xx']}")
        lines.append(f"vsr_http_errors_total {metrics['errors']['total']}")
        lines.append("")
        
        return '\n'.join(lines)


@router.get("")
async def prometheus_metrics():
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus exposition format for scraping.
    """
    try:
        # Get internal metrics
        internal_metrics = get_metrics()
        
        # Add additional system metrics
        additional_metrics = await _get_additional_metrics()
        internal_metrics.update(additional_metrics)
        
        # Format for Prometheus
        prometheus_output = PrometheusMetrics.format_metrics(internal_metrics)
        
        return Response(
            content=prometheus_output,
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )
        
    except Exception as e:
        logger.error("Failed to generate metrics", error=str(e))
        return Response(
            content="# Error generating metrics\n",
            media_type="text/plain",
            status_code=500
        )


async def _get_additional_metrics() -> Dict[str, Any]:
    """
    Get additional system metrics not covered by request middleware.
    
    Returns:
        Dictionary of additional metrics
    """
    additional = {}
    
    # Database connection pool metrics
    try:
        db = await get_db()
        # Add database-specific metrics if available
        additional['database_connected'] = 1
    except Exception:
        additional['database_connected'] = 0
    
    # RabbitMQ queue metrics
    try:
        rabbitmq_client = RabbitMQClient()
        # This would need queue inspection methods
        # queue_depth = await rabbitmq_client.get_queue_depth("vsr.process.q")
        # additional['queue_depth'] = queue_depth
        additional['rabbitmq_connected'] = 1
    except Exception:
        additional['rabbitmq_connected'] = 0
    
    return additional


@router.get("/json")
async def json_metrics():
    """
    JSON format metrics endpoint.
    
    Returns metrics in JSON format for easier debugging and integration.
    """
    try:
        metrics = get_metrics()
        additional_metrics = await _get_additional_metrics()
        metrics.update(additional_metrics)
        return metrics
    except Exception as e:
        logger.error("Failed to generate JSON metrics", error=str(e))
        return {"error": str(e)}
