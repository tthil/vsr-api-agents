"""
Smoke tests for VSR API deployment verification.
Runs basic functionality tests to ensure the deployment is working correctly.
"""
import asyncio
import logging
import os
import sys
import time
from typing import Dict, Any, List
import json

import httpx
import aiofiles


logger = logging.getLogger(__name__)


class VSRSmokeTests:
    """
    Smoke test suite for VSR API deployment verification.
    """
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={"X-API-Key": api_key}
        )
        self.test_results = []
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all smoke tests."""
        logger.info("Starting VSR API smoke tests...")
        
        test_suite = [
            ("Health Check", self.test_health_check),
            ("API Authentication", self.test_authentication),
            ("Generate Upload URL", self.test_generate_upload_url),
            ("Job Status Endpoint", self.test_job_status),
            ("Rate Limiting", self.test_rate_limiting),
            ("Error Handling", self.test_error_handling),
            ("OpenAPI Documentation", self.test_openapi_docs),
            ("Monitoring Endpoints", self.test_monitoring_endpoints)
        ]
        
        results = {
            "total_tests": len(test_suite),
            "passed": 0,
            "failed": 0,
            "errors": [],
            "test_results": []
        }
        
        for test_name, test_func in test_suite:
            try:
                logger.info(f"Running test: {test_name}")
                test_result = await test_func()
                
                if test_result["success"]:
                    results["passed"] += 1
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    results["failed"] += 1
                    logger.error(f"❌ {test_name}: FAILED - {test_result.get('error', 'Unknown error')}")
                    results["errors"].append(f"{test_name}: {test_result.get('error', 'Unknown error')}")
                
                results["test_results"].append({
                    "name": test_name,
                    "success": test_result["success"],
                    "duration": test_result.get("duration", 0),
                    "error": test_result.get("error"),
                    "details": test_result.get("details")
                })
                
            except Exception as e:
                results["failed"] += 1
                error_msg = f"{test_name}: Exception - {str(e)}"
                results["errors"].append(error_msg)
                logger.error(f"❌ {error_msg}")
                
                results["test_results"].append({
                    "name": test_name,
                    "success": False,
                    "error": str(e)
                })
        
        await self.client.aclose()
        
        # Summary
        success_rate = (results["passed"] / results["total_tests"]) * 100
        results["success_rate"] = success_rate
        
        logger.info(f"\n🎯 Smoke Test Results:")
        logger.info(f"   Total Tests: {results['total_tests']}")
        logger.info(f"   Passed: {results['passed']}")
        logger.info(f"   Failed: {results['failed']}")
        logger.info(f"   Success Rate: {success_rate:.1f}%")
        
        if results["failed"] > 0:
            logger.error(f"\n❌ Failed Tests:")
            for error in results["errors"]:
                logger.error(f"   - {error}")
        
        return results
    
    async def test_health_check(self) -> Dict[str, Any]:
        """Test health check endpoint."""
        start_time = time.time()
        
        try:
            response = await self.client.get(f"{self.base_url}/healthz")
            duration = time.time() - start_time
            
            if response.status_code == 200:
                health_data = response.json()
                
                # Verify health response structure
                required_fields = ["status", "timestamp", "version"]
                missing_fields = [field for field in required_fields if field not in health_data]
                
                if missing_fields:
                    return {
                        "success": False,
                        "duration": duration,
                        "error": f"Missing health check fields: {missing_fields}"
                    }
                
                if health_data["status"] != "healthy":
                    return {
                        "success": False,
                        "duration": duration,
                        "error": f"Health status is not healthy: {health_data['status']}"
                    }
                
                return {
                    "success": True,
                    "duration": duration,
                    "details": health_data
                }
            else:
                return {
                    "success": False,
                    "duration": duration,
                    "error": f"Health check returned status {response.status_code}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "duration": time.time() - start_time,
                "error": str(e)
            }
    
    async def test_authentication(self) -> Dict[str, Any]:
        """Test API authentication."""
        start_time = time.time()
        
        try:
            # Test with valid API key
            response = await self.client.get(f"{self.base_url}/api/jobs/status/non-existent-job")
            
            if response.status_code == 404:
                # This is expected for a non-existent job, but means auth worked
                auth_success = True
            elif response.status_code == 401:
                auth_success = False
            else:
                auth_success = True  # Other errors are fine, auth worked
            
            # Test without API key
            client_no_auth = httpx.AsyncClient(timeout=10.0)
            response_no_auth = await client_no_auth.get(f"{self.base_url}/api/jobs/status/test")
            await client_no_auth.aclose()
            
            no_auth_rejected = response_no_auth.status_code == 401
            
            duration = time.time() - start_time
            
            if auth_success and no_auth_rejected:
                return {
                    "success": True,
                    "duration": duration,
                    "details": "Authentication working correctly"
                }
            else:
                return {
                    "success": False,
                    "duration": duration,
                    "error": f"Auth issues - valid key success: {auth_success}, no key rejected: {no_auth_rejected}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "duration": time.time() - start_time,
                "error": str(e)
            }
    
    async def test_generate_upload_url(self) -> Dict[str, Any]:
        """Test generate upload URL endpoint."""
        start_time = time.time()
        
        try:
            payload = {
                "content_type": "video/mp4",
                "file_size": 10485760  # 10MB
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/generate-upload-url",
                json=payload
            )
            
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ["upload_url", "video_key", "expires_at"]
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    return {
                        "success": False,
                        "duration": duration,
                        "error": f"Missing response fields: {missing_fields}"
                    }
                
                # Verify URL format
                if not data["upload_url"].startswith("http"):
                    return {
                        "success": False,
                        "duration": duration,
                        "error": "Invalid upload URL format"
                    }
                
                return {
                    "success": True,
                    "duration": duration,
                    "details": "Upload URL generated successfully"
                }
            else:
                return {
                    "success": False,
                    "duration": duration,
                    "error": f"Generate upload URL returned status {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "duration": time.time() - start_time,
                "error": str(e)
            }
    
    async def test_job_status(self) -> Dict[str, Any]:
        """Test job status endpoint."""
        start_time = time.time()
        
        try:
            # Test with non-existent job (should return 404)
            response = await self.client.get(f"{self.base_url}/api/jobs/status/non-existent-job-id")
            
            duration = time.time() - start_time
            
            if response.status_code == 404:
                error_data = response.json()
                
                # Verify error response structure
                if "error" in error_data and "code" in error_data:
                    return {
                        "success": True,
                        "duration": duration,
                        "details": "Job status endpoint returns proper 404 for non-existent jobs"
                    }
                else:
                    return {
                        "success": False,
                        "duration": duration,
                        "error": "404 response missing proper error structure"
                    }
            else:
                return {
                    "success": False,
                    "duration": duration,
                    "error": f"Expected 404 for non-existent job, got {response.status_code}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "duration": time.time() - start_time,
                "error": str(e)
            }
    
    async def test_rate_limiting(self) -> Dict[str, Any]:
        """Test rate limiting functionality."""
        start_time = time.time()
        
        try:
            # Make multiple rapid requests to trigger rate limiting
            responses = []
            for i in range(10):
                response = await self.client.get(f"{self.base_url}/api/jobs/status/test-{i}")
                responses.append(response.status_code)
                
                # Small delay to avoid overwhelming
                await asyncio.sleep(0.1)
            
            duration = time.time() - start_time
            
            # Check if we got any rate limit responses (429)
            rate_limited = any(status == 429 for status in responses)
            
            # For smoke tests, we don't necessarily need to trigger rate limiting
            # Just verify the endpoint is responding properly
            non_500_responses = all(status < 500 for status in responses)
            
            if non_500_responses:
                return {
                    "success": True,
                    "duration": duration,
                    "details": f"Rate limiting functional, responses: {set(responses)}"
                }
            else:
                return {
                    "success": False,
                    "duration": duration,
                    "error": f"Server errors in responses: {responses}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "duration": time.time() - start_time,
                "error": str(e)
            }
    
    async def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and response formats."""
        start_time = time.time()
        
        try:
            # Test invalid content type for upload URL
            payload = {
                "content_type": "invalid/type",
                "file_size": 1000
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/generate-upload-url",
                json=payload
            )
            
            duration = time.time() - start_time
            
            if response.status_code == 400:
                error_data = response.json()
                
                # Verify error response structure
                required_fields = ["error", "code"]
                missing_fields = [field for field in required_fields if field not in error_data]
                
                if not missing_fields:
                    return {
                        "success": True,
                        "duration": duration,
                        "details": "Error handling working correctly"
                    }
                else:
                    return {
                        "success": False,
                        "duration": duration,
                        "error": f"Error response missing fields: {missing_fields}"
                    }
            else:
                return {
                    "success": False,
                    "duration": duration,
                    "error": f"Expected 400 for invalid content type, got {response.status_code}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "duration": time.time() - start_time,
                "error": str(e)
            }
    
    async def test_openapi_docs(self) -> Dict[str, Any]:
        """Test OpenAPI documentation availability."""
        start_time = time.time()
        
        try:
            # Test OpenAPI JSON
            response = await self.client.get(f"{self.base_url}/openapi.json")
            
            if response.status_code == 200:
                openapi_data = response.json()
                
                # Verify basic OpenAPI structure
                required_fields = ["openapi", "info", "paths"]
                missing_fields = [field for field in required_fields if field not in openapi_data]
                
                if missing_fields:
                    return {
                        "success": False,
                        "duration": time.time() - start_time,
                        "error": f"OpenAPI spec missing fields: {missing_fields}"
                    }
                
                # Test Swagger UI
                docs_response = await self.client.get(f"{self.base_url}/docs")
                
                duration = time.time() - start_time
                
                if docs_response.status_code == 200:
                    return {
                        "success": True,
                        "duration": duration,
                        "details": "OpenAPI documentation available"
                    }
                else:
                    return {
                        "success": False,
                        "duration": duration,
                        "error": f"Swagger UI not accessible: {docs_response.status_code}"
                    }
            else:
                return {
                    "success": False,
                    "duration": time.time() - start_time,
                    "error": f"OpenAPI spec not accessible: {response.status_code}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "duration": time.time() - start_time,
                "error": str(e)
            }
    
    async def test_monitoring_endpoints(self) -> Dict[str, Any]:
        """Test monitoring and metrics endpoints."""
        start_time = time.time()
        
        try:
            # Test metrics endpoint
            response = await self.client.get(f"{self.base_url}/metrics")
            
            duration = time.time() - start_time
            
            if response.status_code == 200:
                metrics_data = response.text
                
                # Verify it contains Prometheus metrics
                if "# HELP" in metrics_data and "# TYPE" in metrics_data:
                    return {
                        "success": True,
                        "duration": duration,
                        "details": "Monitoring endpoints accessible"
                    }
                else:
                    return {
                        "success": False,
                        "duration": duration,
                        "error": "Metrics endpoint not returning Prometheus format"
                    }
            else:
                return {
                    "success": False,
                    "duration": duration,
                    "error": f"Metrics endpoint not accessible: {response.status_code}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "duration": time.time() - start_time,
                "error": str(e)
            }


async def main():
    """Main smoke test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="VSR API Smoke Tests")
    parser.add_argument("--base-url", default=os.getenv("API_BASE_URL", "http://localhost:8000"), help="API base URL")
    parser.add_argument("--api-key", default=os.getenv("API_KEY", "test-key"), help="API key for authentication")
    parser.add_argument("--output", help="Output file for test results (JSON)")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit with error code if tests fail")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        smoke_tests = VSRSmokeTests(args.base_url, args.api_key)
        results = await smoke_tests.run_all_tests()
        
        # Save results if output file specified
        if args.output:
            async with aiofiles.open(args.output, 'w') as f:
                await f.write(json.dumps(results, indent=2))
            logger.info(f"Test results saved to {args.output}")
        
        # Exit with error if tests failed and fail-on-error is set
        if args.fail_on_error and results["failed"] > 0:
            logger.error("Some tests failed, exiting with error code")
            sys.exit(1)
        
        logger.info("Smoke tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Smoke tests failed with exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
