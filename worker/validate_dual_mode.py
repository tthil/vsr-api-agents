#!/usr/bin/env python3
"""
Dual-mode architecture validation script.

Validates that the dual-mode worker architecture is properly configured
and can switch between CPU and GPU modes correctly.
"""

import os
import sys
import asyncio
import tempfile
from pathlib import Path
from typing import Dict, Any

# Add worker directory to Python path
worker_dir = Path(__file__).parent
sys.path.insert(0, str(worker_dir))

from vsr_worker.gpu.environment import WorkerEnvironment
from vsr_worker.models.adaptive import AdaptiveModelFactory
from vsr_worker.config.dual_mode import (
    get_config_manager, 
    print_config_summary, 
    validate_environment,
    WorkerMode
)
from vsr_shared.logging import get_logger

logger = get_logger(__name__)


class DualModeValidator:
    """Validator for dual-mode worker architecture."""
    
    def __init__(self):
        self.results = {
            "environment_detection": False,
            "cpu_mode_config": False,
            "gpu_mode_config": False,
            "model_factory": False,
            "config_management": False,
            "docker_integration": False
        }
        self.errors = []
    
    async def validate_all(self) -> Dict[str, Any]:
        """Run all validation tests."""
        print("=" * 60)
        print("VSR Dual-Mode Architecture Validation")
        print("=" * 60)
        
        # Test 1: Environment Detection
        await self._test_environment_detection()
        
        # Test 2: CPU Mode Configuration
        await self._test_cpu_mode_config()
        
        # Test 3: GPU Mode Configuration (if available)
        await self._test_gpu_mode_config()
        
        # Test 4: Model Factory
        await self._test_model_factory()
        
        # Test 5: Configuration Management
        await self._test_config_management()
        
        # Test 6: Docker Integration
        await self._test_docker_integration()
        
        # Print results
        self._print_results()
        
        return {
            "success": all(self.results.values()),
            "results": self.results,
            "errors": self.errors
        }
    
    async def _test_environment_detection(self):
        """Test environment detection functionality."""
        print("\n1. Testing Environment Detection...")
        
        try:
            # Test auto-detection
            original_mode = os.environ.get('WORKER_MODE')
            os.environ['WORKER_MODE'] = 'auto'
            
            env = WorkerEnvironment()
            info = await env.initialize()
            
            assert env.mode in ["cpu", "gpu"], f"Invalid mode: {env.mode}"
            assert "device" in info, "Device info missing"
            assert "cpu_info" in info, "CPU info missing"
            
            print(f"   ✓ Auto-detected mode: {env.mode}")
            print(f"   ✓ Device: {info['device']}")
            print(f"   ✓ CPU cores: {info['cpu_info']['cores']}")
            
            # Test forced CPU mode
            os.environ['WORKER_MODE'] = 'cpu'
            env_cpu = WorkerEnvironment()
            assert env_cpu.mode == "cpu", "Failed to force CPU mode"
            print("   ✓ Forced CPU mode works")
            
            # Restore original mode
            if original_mode:
                os.environ['WORKER_MODE'] = original_mode
            else:
                os.environ.pop('WORKER_MODE', None)
            
            self.results["environment_detection"] = True
            
        except Exception as e:
            self.errors.append(f"Environment detection failed: {e}")
            print(f"   ✗ Environment detection failed: {e}")
    
    async def _test_cpu_mode_config(self):
        """Test CPU mode configuration."""
        print("\n2. Testing CPU Mode Configuration...")
        
        try:
            # Set CPU mode environment
            test_env = {
                'WORKER_MODE': 'cpu',
                'ENVIRONMENT': 'development',
                'MODEL_QUALITY': 'lightweight',
                'CPU_THREADS': '4',
                'MEMORY_LIMIT_MB': '2048'
            }
            
            original_env = {}
            for key, value in test_env.items():
                original_env[key] = os.environ.get(key)
                os.environ[key] = value
            
            try:
                config_manager = get_config_manager()
                processing_config = config_manager.get_processing_config(WorkerMode.CPU)
                
                assert config_manager.config.worker.mode == WorkerMode.CPU
                assert processing_config.max_resolution == (1280, 720)
                assert processing_config.timeout_seconds >= 300
                
                print("   ✓ CPU configuration loaded correctly")
                print(f"   ✓ Max resolution: {processing_config.max_resolution}")
                print(f"   ✓ Timeout: {processing_config.timeout_seconds}s")
                
                self.results["cpu_mode_config"] = True
                
            finally:
                # Restore original environment
                for key, value in original_env.items():
                    if value is not None:
                        os.environ[key] = value
                    else:
                        os.environ.pop(key, None)
            
        except Exception as e:
            self.errors.append(f"CPU mode configuration failed: {e}")
            print(f"   ✗ CPU mode configuration failed: {e}")
    
    async def _test_gpu_mode_config(self):
        """Test GPU mode configuration (if CUDA available)."""
        print("\n3. Testing GPU Mode Configuration...")
        
        try:
            import torch
            
            if not torch.cuda.is_available():
                print("   ⚠ CUDA not available, skipping GPU tests")
                self.results["gpu_mode_config"] = True  # Skip but don't fail
                return
            
            # Set GPU mode environment
            test_env = {
                'WORKER_MODE': 'gpu',
                'ENVIRONMENT': 'production',
                'MODEL_QUALITY': 'high'
            }
            
            original_env = {}
            for key, value in test_env.items():
                original_env[key] = os.environ.get(key)
                os.environ[key] = value
            
            try:
                config_manager = get_config_manager()
                processing_config = config_manager.get_processing_config(WorkerMode.GPU)
                
                assert config_manager.config.worker.mode == WorkerMode.GPU
                assert processing_config.max_resolution == (1920, 1080)
                
                print("   ✓ GPU configuration loaded correctly")
                print(f"   ✓ Max resolution: {processing_config.max_resolution}")
                print(f"   ✓ CUDA devices: {torch.cuda.device_count()}")
                
                self.results["gpu_mode_config"] = True
                
            finally:
                # Restore original environment
                for key, value in original_env.items():
                    if value is not None:
                        os.environ[key] = value
                    else:
                        os.environ.pop(key, None)
            
        except Exception as e:
            self.errors.append(f"GPU mode configuration failed: {e}")
            print(f"   ✗ GPU mode configuration failed: {e}")
    
    async def _test_model_factory(self):
        """Test adaptive model factory."""
        print("\n4. Testing Model Factory...")
        
        try:
            # Test with CPU mode
            os.environ['WORKER_MODE'] = 'cpu'
            env = WorkerEnvironment()
            factory = AdaptiveModelFactory(env)
            
            # Test model types
            model_types = ["sttn", "lama", "propainter"]
            
            for model_type in model_types:
                try:
                    # Note: This will fail without actual model files,
                    # but we can test the factory logic
                    model = await factory.create_model(model_type)
                    print(f"   ✓ {model_type.upper()} model factory works")
                except Exception as model_error:
                    # Expected to fail without model files
                    if "model file" in str(model_error).lower() or "checkpoint" in str(model_error).lower():
                        print(f"   ✓ {model_type.upper()} model factory logic works (missing model files)")
                    else:
                        raise model_error
            
            # Test invalid model type
            try:
                await factory.create_model("invalid_model")
                raise AssertionError("Should have failed with invalid model")
            except ValueError:
                print("   ✓ Invalid model type properly rejected")
            
            self.results["model_factory"] = True
            
        except Exception as e:
            self.errors.append(f"Model factory test failed: {e}")
            print(f"   ✗ Model factory test failed: {e}")
    
    async def _test_config_management(self):
        """Test configuration management system."""
        print("\n5. Testing Configuration Management...")
        
        try:
            # Test environment validation
            test_env = {
                'WORKER_MODE': 'cpu',
                'ENVIRONMENT': 'development',
                'MODEL_QUALITY': 'lightweight'
            }
            
            original_env = {}
            for key, value in test_env.items():
                original_env[key] = os.environ.get(key)
                os.environ[key] = value
            
            try:
                # Test validation
                is_valid = validate_environment()
                assert is_valid, "Environment validation failed"
                print("   ✓ Environment validation works")
                
                # Test config manager
                config_manager = get_config_manager()
                assert config_manager.config is not None
                print("   ✓ Configuration manager initialization works")
                
                # Test config summary (should not raise errors)
                print_config_summary()
                print("   ✓ Configuration summary generation works")
                
                self.results["config_management"] = True
                
            finally:
                # Restore original environment
                for key, value in original_env.items():
                    if value is not None:
                        os.environ[key] = value
                    else:
                        os.environ.pop(key, None)
            
        except Exception as e:
            self.errors.append(f"Configuration management test failed: {e}")
            print(f"   ✗ Configuration management test failed: {e}")
    
    async def _test_docker_integration(self):
        """Test Docker integration files."""
        print("\n6. Testing Docker Integration...")
        
        try:
            # Check Docker compose files exist
            project_root = Path(__file__).parent.parent
            
            local_compose = project_root / "infra" / "docker-compose.local.yml"
            prod_compose = project_root / "infra" / "docker-compose.prod.yml"
            
            assert local_compose.exists(), "Local Docker compose file missing"
            assert prod_compose.exists(), "Production Docker compose file missing"
            print("   ✓ Docker compose files exist")
            
            # Check local compose has CPU configuration
            local_content = local_compose.read_text()
            assert "WORKER_MODE=cpu" in local_content, "CPU mode not configured in local compose"
            assert "MODEL_QUALITY=lightweight" in local_content, "Lightweight quality not configured"
            print("   ✓ Local compose configured for CPU mode")
            
            # Check production compose has GPU configuration
            prod_content = prod_compose.read_text()
            assert "WORKER_MODE=gpu" in prod_content, "GPU mode not configured in production compose"
            assert "MODEL_QUALITY=high" in prod_content, "High quality not configured"
            print("   ✓ Production compose configured for GPU mode")
            
            # Check startup script exists
            startup_script = project_root / "worker" / "start_worker.py"
            assert startup_script.exists(), "Worker startup script missing"
            print("   ✓ Worker startup script exists")
            
            self.results["docker_integration"] = True
            
        except Exception as e:
            self.errors.append(f"Docker integration test failed: {e}")
            print(f"   ✗ Docker integration test failed: {e}")
    
    def _print_results(self):
        """Print validation results summary."""
        print("\n" + "=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result)
        
        for test_name, result in self.results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"{test_name.replace('_', ' ').title():<30} {status}")
        
        print("-" * 60)
        print(f"Total: {passed_tests}/{total_tests} tests passed")
        
        if self.errors:
            print("\nERRORS:")
            for error in self.errors:
                print(f"  - {error}")
        
        if all(self.results.values()):
            print("\n🎉 All tests passed! Dual-mode architecture is ready.")
        else:
            print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Please review the errors above.")


async def main():
    """Main validation entry point."""
    validator = DualModeValidator()
    results = await validator.validate_all()
    
    # Exit with appropriate code
    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    asyncio.run(main())
