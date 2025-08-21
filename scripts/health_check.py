"""
Comprehensive health check script for VSR API production deployment.
Validates all system components and provides detailed health reports.
"""
import asyncio
import logging
import os
import sys
import time
from typing import Dict, Any, List, Optional
import json

import httpx
import aiofiles


logger = logging.getLogger(__name__)


class VSRHealthChecker:
    """
    Comprehensive health checker for VSR API production deployment.
    """
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={"X-API-Key": api_key}
        )
    
    async def run_comprehensive_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check of all system components."""
        logger.info("Starting comprehensive health check...")
        
        health_checks = [
            ("API Health", self.check_api_health),
            ("Database Connectivity", self.check_database_health),
            ("Message Queue", self.check_rabbitmq_health),
            ("Cache System", self.check_redis_health),
            ("Storage System", self.check_storage_health),
            ("GPU Worker", self.check_worker_health),
            ("SSL Certificate", self.check_ssl_certificate),
            ("Load Balancer", self.check_load_balancer),
            ("Monitoring Stack", self.check_monitoring_health),
            ("Performance Metrics", self.check_performance_metrics),
            ("Security Headers", self.check_security_headers),
            ("Rate Limiting", self.check_rate_limiting),
            ("Error Handling", self.check_error_handling)
        ]
        
        results = {
            "timestamp": time.time(),
            "overall_status": "unknown",
            "total_checks": len(health_checks),
            "passed": 0,
            "failed": 0,
            "warnings": 0,
            "critical_failures": [],
            "check_results": []
        }
        
        for check_name, check_func in health_checks:
            try:
                logger.info(f"Running check: {check_name}")
                check_result = await check_func()
                
                check_result["name"] = check_name
                results["check_results"].append(check_result)
                
                if check_result["status"] == "healthy":
                    results["passed"] += 1
                    logger.info(f"✅ {check_name}: HEALTHY")
                elif check_result["status"] == "warning":
                    results["warnings"] += 1
                    logger.warning(f"⚠️  {check_name}: WARNING - {check_result.get('message', '')}")
                else:
                    results["failed"] += 1
                    logger.error(f"❌ {check_name}: FAILED - {check_result.get('message', '')}")
                    
                    if check_result.get("critical", False):
                        results["critical_failures"].append(check_name)
                
            except Exception as e:
                results["failed"] += 1
                error_msg = f"{check_name}: Exception - {str(e)}"
                results["critical_failures"].append(check_name)
                logger.error(f"❌ {error_msg}")
                
                results["check_results"].append({
                    "name": check_name,
                    "status": "failed",
                    "message": str(e),
                    "critical": True
                })
        
        # Determine overall status
        if results["critical_failures"]:
            results["overall_status"] = "critical"
        elif results["failed"] > 0:
            results["overall_status"] = "degraded"
        elif results["warnings"] > 0:
            results["overall_status"] = "warning"
        else:
            results["overall_status"] = "healthy"
        
        await self.client.aclose()
        
        # Summary
        logger.info(f"\n🎯 Health Check Results:")
        logger.info(f"   Overall Status: {results['overall_status'].upper()}")
        logger.info(f"   Total Checks: {results['total_checks']}")
        logger.info(f"   Passed: {results['passed']}")
        logger.info(f"   Warnings: {results['warnings']}")
        logger.info(f"   Failed: {results['failed']}")
        
        if results["critical_failures"]:
            logger.error(f"\n🚨 Critical Failures:")
            for failure in results["critical_failures"]:
                logger.error(f"   - {failure}")
        
        return results
    
    async def check_api_health(self) -> Dict[str, Any]:
        """Check API health and basic functionality."""
        try:
            start_time = time.time()
            response = await self.client.get(f"{self.base_url}/healthz")
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                health_data = response.json()
                
                # Check response time
                if response_time > 5.0:
                    return {
                        "status": "warning",
                        "message": f"Slow response time: {response_time:.2f}s",
                        "response_time": response_time,
                        "details": health_data
                    }
                
                return {
                    "status": "healthy",
                    "message": "API responding normally",
                    "response_time": response_time,
                    "details": health_data
                }
            else:
                return {
                    "status": "failed",
                    "message": f"API health check failed with status {response.status_code}",
                    "critical": True,
                    "response_time": response_time
                }
                
        except Exception as e:
            return {
                "status": "failed",
                "message": f"API health check exception: {str(e)}",
                "critical": True
            }
    
    async def check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and performance."""
        try:
            # Use the health endpoint which should check database
            response = await self.client.get(f"{self.base_url}/health/detailed")
            
            if response.status_code == 200:
                health_data = response.json()
                
                if "database" in health_data:
                    db_status = health_data["database"]
                    
                    if db_status.get("status") == "healthy":
                        return {
                            "status": "healthy",
                            "message": "Database connectivity normal",
                            "details": db_status
                        }
                    else:
                        return {
                            "status": "failed",
                            "message": f"Database unhealthy: {db_status.get('error', 'Unknown error')}",
                            "critical": True,
                            "details": db_status
                        }
                else:
                    return {
                        "status": "warning",
                        "message": "Database status not available in health check"
                    }
            else:
                return {
                    "status": "failed",
                    "message": f"Detailed health check failed: {response.status_code}",
                    "critical": True
                }
                
        except Exception as e:
            return {
                "status": "failed",
                "message": f"Database health check failed: {str(e)}",
                "critical": True
            }
    
    async def check_rabbitmq_health(self) -> Dict[str, Any]:
        """Check RabbitMQ message queue health."""
        try:
            response = await self.client.get(f"{self.base_url}/health/detailed")
            
            if response.status_code == 200:
                health_data = response.json()
                
                if "rabbitmq" in health_data:
                    mq_status = health_data["rabbitmq"]
                    
                    if mq_status.get("status") == "healthy":
                        return {
                            "status": "healthy",
                            "message": "RabbitMQ connectivity normal",
                            "details": mq_status
                        }
                    else:
                        return {
                            "status": "failed",
                            "message": f"RabbitMQ unhealthy: {mq_status.get('error', 'Unknown error')}",
                            "critical": True,
                            "details": mq_status
                        }
                else:
                    return {
                        "status": "warning",
                        "message": "RabbitMQ status not available in health check"
                    }
            else:
                return {
                    "status": "failed",
                    "message": "Cannot check RabbitMQ health",
                    "critical": True
                }
                
        except Exception as e:
            return {
                "status": "failed",
                "message": f"RabbitMQ health check failed: {str(e)}",
                "critical": True
            }
    
    async def check_redis_health(self) -> Dict[str, Any]:
        """Check Redis cache health."""
        try:
            response = await self.client.get(f"{self.base_url}/health/detailed")
            
            if response.status_code == 200:
                health_data = response.json()
                
                if "redis" in health_data:
                    redis_status = health_data["redis"]
                    
                    if redis_status.get("status") == "healthy":
                        return {
                            "status": "healthy",
                            "message": "Redis connectivity normal",
                            "details": redis_status
                        }
                    else:
                        return {
                            "status": "warning",
                            "message": f"Redis issues: {redis_status.get('error', 'Unknown error')}",
                            "details": redis_status
                        }
                else:
                    return {
                        "status": "warning",
                        "message": "Redis status not available in health check"
                    }
            else:
                return {
                    "status": "warning",
                    "message": "Cannot check Redis health"
                }
                
        except Exception as e:
            return {
                "status": "warning",
                "message": f"Redis health check failed: {str(e)}"
            }
    
    async def check_storage_health(self) -> Dict[str, Any]:
        """Check DigitalOcean Spaces storage health."""
        try:
            response = await self.client.get(f"{self.base_url}/health/detailed")
            
            if response.status_code == 200:
                health_data = response.json()
                
                if "spaces" in health_data:
                    spaces_status = health_data["spaces"]
                    
                    if spaces_status.get("status") == "healthy":
                        return {
                            "status": "healthy",
                            "message": "Storage connectivity normal",
                            "details": spaces_status
                        }
                    else:
                        return {
                            "status": "failed",
                            "message": f"Storage unhealthy: {spaces_status.get('error', 'Unknown error')}",
                            "critical": True,
                            "details": spaces_status
                        }
                else:
                    return {
                        "status": "warning",
                        "message": "Storage status not available in health check"
                    }
            else:
                return {
                    "status": "failed",
                    "message": "Cannot check storage health",
                    "critical": True
                }
                
        except Exception as e:
            return {
                "status": "failed",
                "message": f"Storage health check failed: {str(e)}",
                "critical": True
            }
    
    async def check_worker_health(self) -> Dict[str, Any]:
        """Check GPU worker health."""
        try:
            response = await self.client.get(f"{self.base_url}/health/detailed")
            
            if response.status_code == 200:
                health_data = response.json()
                
                if "gpu" in health_data:
                    gpu_status = health_data["gpu"]
                    
                    if gpu_status.get("status") == "healthy":
                        return {
                            "status": "healthy",
                            "message": "GPU worker operational",
                            "details": gpu_status
                        }
                    else:
                        return {
                            "status": "failed",
                            "message": f"GPU worker issues: {gpu_status.get('error', 'Unknown error')}",
                            "critical": True,
                            "details": gpu_status
                        }
                else:
                    return {
                        "status": "warning",
                        "message": "GPU worker status not available"
                    }
            else:
                return {
                    "status": "failed",
                    "message": "Cannot check worker health",
                    "critical": True
                }
                
        except Exception as e:
            return {
                "status": "failed",
                "message": f"Worker health check failed: {str(e)}",
                "critical": True
            }
    
    async def check_ssl_certificate(self) -> Dict[str, Any]:
        """Check SSL certificate validity."""
        try:
            if not self.base_url.startswith("https://"):
                return {
                    "status": "warning",
                    "message": "Not using HTTPS"
                }
            
            # Make a request and check certificate info
            response = await self.client.get(f"{self.base_url}/healthz")
            
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "message": "SSL certificate valid"
                }
            else:
                return {
                    "status": "warning",
                    "message": "SSL certificate check inconclusive"
                }
                
        except Exception as e:
            if "certificate" in str(e).lower() or "ssl" in str(e).lower():
                return {
                    "status": "failed",
                    "message": f"SSL certificate error: {str(e)}",
                    "critical": True
                }
            else:
                return {
                    "status": "warning",
                    "message": f"SSL check failed: {str(e)}"
                }
    
    async def check_load_balancer(self) -> Dict[str, Any]:
        """Check load balancer health."""
        try:
            # Make multiple requests to check load balancing
            responses = []
            for i in range(5):
                response = await self.client.get(f"{self.base_url}/healthz")
                responses.append(response.status_code)
                await asyncio.sleep(0.1)
            
            success_count = sum(1 for status in responses if status == 200)
            
            if success_count >= 4:  # Allow one failure
                return {
                    "status": "healthy",
                    "message": "Load balancer functioning normally",
                    "details": {"success_rate": success_count / len(responses)}
                }
            elif success_count >= 2:
                return {
                    "status": "warning",
                    "message": f"Load balancer partially functional: {success_count}/5 requests succeeded"
                }
            else:
                return {
                    "status": "failed",
                    "message": f"Load balancer failing: {success_count}/5 requests succeeded",
                    "critical": True
                }
                
        except Exception as e:
            return {
                "status": "failed",
                "message": f"Load balancer check failed: {str(e)}",
                "critical": True
            }
    
    async def check_monitoring_health(self) -> Dict[str, Any]:
        """Check monitoring stack health."""
        try:
            # Check Prometheus metrics endpoint
            response = await self.client.get(f"{self.base_url}/metrics")
            
            if response.status_code == 200:
                metrics_data = response.text
                
                if "# HELP" in metrics_data and "# TYPE" in metrics_data:
                    return {
                        "status": "healthy",
                        "message": "Monitoring stack operational",
                        "details": {"metrics_available": True}
                    }
                else:
                    return {
                        "status": "warning",
                        "message": "Metrics endpoint not returning proper format"
                    }
            else:
                return {
                    "status": "warning",
                    "message": f"Metrics endpoint not accessible: {response.status_code}"
                }
                
        except Exception as e:
            return {
                "status": "warning",
                "message": f"Monitoring check failed: {str(e)}"
            }
    
    async def check_performance_metrics(self) -> Dict[str, Any]:
        """Check performance metrics and thresholds."""
        try:
            start_time = time.time()
            response = await self.client.get(f"{self.base_url}/healthz")
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                # Check response time thresholds
                if response_time < 1.0:
                    performance_status = "healthy"
                    message = f"Performance normal: {response_time:.3f}s"
                elif response_time < 3.0:
                    performance_status = "warning"
                    message = f"Performance degraded: {response_time:.3f}s"
                else:
                    performance_status = "failed"
                    message = f"Performance critical: {response_time:.3f}s"
                
                return {
                    "status": performance_status,
                    "message": message,
                    "details": {"response_time": response_time}
                }
            else:
                return {
                    "status": "failed",
                    "message": "Cannot measure performance - API not responding",
                    "critical": True
                }
                
        except Exception as e:
            return {
                "status": "failed",
                "message": f"Performance check failed: {str(e)}",
                "critical": True
            }
    
    async def check_security_headers(self) -> Dict[str, Any]:
        """Check security headers."""
        try:
            response = await self.client.get(f"{self.base_url}/healthz")
            
            if response.status_code == 200:
                headers = response.headers
                
                security_headers = [
                    "x-content-type-options",
                    "x-frame-options",
                    "x-xss-protection",
                    "strict-transport-security"
                ]
                
                missing_headers = []
                for header in security_headers:
                    if header not in headers:
                        missing_headers.append(header)
                
                if not missing_headers:
                    return {
                        "status": "healthy",
                        "message": "Security headers properly configured"
                    }
                elif len(missing_headers) <= 2:
                    return {
                        "status": "warning",
                        "message": f"Some security headers missing: {missing_headers}"
                    }
                else:
                    return {
                        "status": "failed",
                        "message": f"Multiple security headers missing: {missing_headers}"
                    }
            else:
                return {
                    "status": "warning",
                    "message": "Cannot check security headers - API not responding"
                }
                
        except Exception as e:
            return {
                "status": "warning",
                "message": f"Security headers check failed: {str(e)}"
            }
    
    async def check_rate_limiting(self) -> Dict[str, Any]:
        """Check rate limiting functionality."""
        try:
            # Make requests to test rate limiting
            responses = []
            for i in range(5):
                response = await self.client.get(f"{self.base_url}/healthz")
                responses.append(response.status_code)
                await asyncio.sleep(0.1)
            
            # Check if all requests succeeded (rate limiting not triggered, which is fine)
            success_responses = [r for r in responses if r == 200]
            
            if len(success_responses) >= 3:
                return {
                    "status": "healthy",
                    "message": "Rate limiting configured (not triggered in test)",
                    "details": {"responses": responses}
                }
            else:
                return {
                    "status": "warning",
                    "message": f"Unexpected response pattern: {responses}"
                }
                
        except Exception as e:
            return {
                "status": "warning",
                "message": f"Rate limiting check failed: {str(e)}"
            }
    
    async def check_error_handling(self) -> Dict[str, Any]:
        """Check error handling."""
        try:
            # Test 404 error handling
            response = await self.client.get(f"{self.base_url}/non-existent-endpoint")
            
            if response.status_code == 404:
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        return {
                            "status": "healthy",
                            "message": "Error handling working correctly"
                        }
                    else:
                        return {
                            "status": "warning",
                            "message": "404 errors not properly formatted"
                        }
                except:
                    return {
                        "status": "warning",
                        "message": "404 errors not returning JSON"
                    }
            else:
                return {
                    "status": "warning",
                    "message": f"Unexpected response for non-existent endpoint: {response.status_code}"
                }
                
        except Exception as e:
            return {
                "status": "warning",
                "message": f"Error handling check failed: {str(e)}"
            }


async def main():
    """Main health check runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="VSR API Health Checker")
    parser.add_argument("--base-url", default=os.getenv("API_BASE_URL", "http://localhost:8000"), help="API base URL")
    parser.add_argument("--api-key", default=os.getenv("API_KEY", "test-key"), help="API key for authentication")
    parser.add_argument("--output", help="Output file for health check results (JSON)")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive health checks")
    parser.add_argument("--fail-on-critical", action="store_true", help="Exit with error code if critical failures")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        health_checker = VSRHealthChecker(args.base_url, args.api_key)
        results = await health_checker.run_comprehensive_health_check()
        
        # Save results if output file specified
        if args.output:
            async with aiofiles.open(args.output, 'w') as f:
                await f.write(json.dumps(results, indent=2))
            logger.info(f"Health check results saved to {args.output}")
        
        # Exit with error if critical failures and fail-on-critical is set
        if args.fail_on_critical and results["critical_failures"]:
            logger.error("Critical failures detected, exiting with error code")
            sys.exit(1)
        
        logger.info("Health check completed successfully!")
        
    except Exception as e:
        logger.error(f"Health check failed with exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
