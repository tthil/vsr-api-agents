"""GPU environment setup and management for VSR worker."""

import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple
import torch
import psutil

from vsr_shared.logging import get_logger

logger = get_logger(__name__)


class WorkerEnvironment:
    """Adaptive worker environment manager supporting both CPU and GPU modes."""

    def __init__(self):
        """Initialize worker environment manager."""
        self.device = None
        self.mode = None
        self.cuda_available = False
        self.cpu_info = {}
        self.gpu_info = {}
        self.memory_info = {}
        self.config = {}

    async def initialize(self) -> Dict[str, any]:
        """
        Initialize worker environment with automatic CPU/GPU detection.

        Returns:
            Dict containing worker environment information
        """
        logger.info("Initializing worker environment...")

        try:
            # Detect worker mode
            self.mode = self._detect_worker_mode()
            logger.info(f"Worker mode detected: {self.mode}")

            if self.mode == "gpu":
                return await self._initialize_gpu_mode()
            else:
                return await self._initialize_cpu_mode()

        except Exception as e:
            logger.error(f"Environment initialization failed: {e}")
            # Fallback to CPU mode
            logger.info("Falling back to CPU mode...")
            self.mode = "cpu"
            return await self._initialize_cpu_mode()

    def _detect_worker_mode(self) -> str:
        """
        Detect optimal worker mode based on environment and hardware.
        
        Returns:
            Worker mode: 'cpu', 'gpu', or 'auto'
        """
        # Check environment variable override
        worker_mode = os.getenv("WORKER_MODE", "auto").lower()
        
        if worker_mode in ["cpu", "gpu"]:
            logger.info(f"Worker mode explicitly set to: {worker_mode}")
            return worker_mode
        
        # Auto-detection logic
        self.cuda_available = torch.cuda.is_available()
        is_production = os.getenv("ENVIRONMENT", "development").lower() == "production"
        
        if self.cuda_available and is_production:
            return "gpu"
        elif self.cuda_available and not is_production:
            # Local development with GPU available - check preference
            prefer_gpu = os.getenv("PREFER_GPU_LOCAL", "false").lower() == "true"
            return "gpu" if prefer_gpu else "cpu"
        else:
            return "cpu"

    async def _initialize_gpu_mode(self) -> Dict[str, any]:
        """Initialize GPU processing mode."""
        logger.info("Initializing GPU mode...")
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("GPU mode requested but CUDA is not available")

        # Get GPU device
        self.device = torch.device("cuda:0")
        gpu_count = torch.cuda.device_count()
        
        logger.info(f"CUDA available: {self.cuda_available}")
        logger.info(f"GPU count: {gpu_count}")
        logger.info(f"Current device: {self.device}")

        # Get GPU information
        self.gpu_info = await self._get_gpu_info()
        self.memory_info = await self._get_memory_info()

        # Validate GPU capabilities
        await self._validate_gpu_capabilities()

        # Set optimal GPU settings
        await self._configure_gpu_settings()

        environment_info = {
            "mode": "gpu",
            "cuda_available": self.cuda_available,
            "device": str(self.device),
            "gpu_count": gpu_count,
            "gpu_info": self.gpu_info,
            "memory_info": self.memory_info,
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }

        logger.info("GPU environment initialized successfully")
        return environment_info

    async def _initialize_cpu_mode(self) -> Dict[str, any]:
        """Initialize CPU processing mode."""
        logger.info("Initializing CPU mode...")
        
        # Set CPU device
        self.device = torch.device("cpu")
        self.cuda_available = False
        
        # Get CPU information
        self.cpu_info = await self._get_cpu_info()
        self.memory_info = await self._get_cpu_memory_info()
        
        # Configure CPU settings
        await self._configure_cpu_settings()
        
        environment_info = {
            "mode": "cpu",
            "cuda_available": False,
            "device": str(self.device),
            "cpu_info": self.cpu_info,
            "memory_info": self.memory_info,
            "pytorch_version": torch.__version__,
            "cpu_threads": torch.get_num_threads(),
        }
        
        logger.info("CPU environment initialized successfully")
        return environment_info

    async def _get_cpu_info(self) -> Dict[str, any]:
        """Get CPU information."""
        try:
            import platform
            
            cpu_info = {
                "processor": platform.processor(),
                "architecture": platform.machine(),
                "cpu_count": psutil.cpu_count(logical=False),
                "logical_cpu_count": psutil.cpu_count(logical=True),
                "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            }
            
            return cpu_info
        except Exception as e:
            logger.warning(f"Could not get CPU info: {e}")
            return {}

    async def _get_cpu_memory_info(self) -> Dict[str, any]:
        """Get CPU memory information."""
        try:
            memory = psutil.virtual_memory()
            return {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_percent": memory.percent,
            }
        except Exception as e:
            logger.warning(f"Could not get memory info: {e}")
            return {}

    async def _configure_cpu_settings(self):
        """Configure optimal CPU settings."""
        try:
            # Set number of threads for CPU processing
            cpu_threads = int(os.getenv("CPU_THREADS", str(psutil.cpu_count(logical=True))))
            torch.set_num_threads(cpu_threads)
            
            # Set CPU optimization flags
            torch.set_num_interop_threads(1)  # Reduce thread contention
            
            logger.info(f"CPU configured with {cpu_threads} threads")
        except Exception as e:
            logger.warning(f"CPU configuration warning: {e}")

    async def _get_gpu_info(self) -> Dict[str, any]:
        """Get detailed GPU information."""
        try:
            gpu_info = {}
            
            if torch.cuda.is_available():
                device_props = torch.cuda.get_device_properties(0)
                gpu_info = {
                    "name": device_props.name,
                    "major": device_props.major,
                    "minor": device_props.minor,
                    "total_memory": device_props.total_memory,
                    "multi_processor_count": device_props.multi_processor_count,
                    "max_threads_per_multi_processor": device_props.max_threads_per_multi_processor,
                    "max_shared_memory_per_block": device_props.max_shared_memory_per_block,
                }
                
                # Check if it's H100 or similar high-end GPU
                if "H100" in device_props.name:
                    gpu_info["is_h100"] = True
                    gpu_info["optimization_level"] = "h100"
                elif "A100" in device_props.name:
                    gpu_info["is_a100"] = True
                    gpu_info["optimization_level"] = "a100"
                else:
                    gpu_info["optimization_level"] = "standard"

            return gpu_info

        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")
            return {}

    async def _get_memory_info(self) -> Dict[str, any]:
        """Get GPU memory information."""
        try:
            if not torch.cuda.is_available():
                return {}

            memory_info = {
                "total": torch.cuda.get_device_properties(0).total_memory,
                "allocated": torch.cuda.memory_allocated(0),
                "cached": torch.cuda.memory_reserved(0),
            }
            
            memory_info["free"] = memory_info["total"] - memory_info["allocated"]
            memory_info["utilization"] = memory_info["allocated"] / memory_info["total"] * 100

            return memory_info

        except Exception as e:
            logger.error(f"Error getting memory info: {e}")
            return {}

    async def _validate_gpu_capabilities(self) -> None:
        """Validate GPU capabilities for model requirements."""
        if not self.gpu_info:
            raise RuntimeError("GPU information not available")

        # Check minimum compute capability (7.0 for modern models)
        compute_capability = f"{self.gpu_info.get('major', 0)}.{self.gpu_info.get('minor', 0)}"
        min_compute = 7.0
        current_compute = float(compute_capability)
        
        if current_compute < min_compute:
            raise RuntimeError(
                f"GPU compute capability {compute_capability} is below minimum {min_compute}"
            )

        # Check minimum memory (8GB for model loading)
        total_memory_gb = self.gpu_info.get("total_memory", 0) / (1024**3)
        min_memory_gb = 8.0
        
        if total_memory_gb < min_memory_gb:
            raise RuntimeError(
                f"GPU memory {total_memory_gb:.1f}GB is below minimum {min_memory_gb}GB"
            )

        logger.info(f"GPU validation passed: {compute_capability} compute, {total_memory_gb:.1f}GB memory")

    async def _configure_gpu_settings(self) -> None:
        """Configure optimal GPU settings."""
        try:
            # Enable TensorFloat-32 (TF32) for H100/A100
            if self.gpu_info.get("is_h100") or self.gpu_info.get("is_a100"):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("Enabled TF32 for high-end GPU")

            # Set memory fraction to prevent OOM
            memory_fraction = 0.9  # Use 90% of GPU memory
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
            
            # Enable cuDNN benchmarking for consistent input sizes
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

            # Set optimal number of threads
            optimal_threads = min(torch.get_num_threads(), psutil.cpu_count())
            torch.set_num_threads(optimal_threads)

            logger.info(f"GPU settings configured: memory_fraction={memory_fraction}, threads={optimal_threads}")

        except Exception as e:
            logger.error(f"Error configuring GPU settings: {e}")

    async def health_check(self) -> Dict[str, any]:
        """
        Perform GPU health check.

        Returns:
            Dict containing health status and metrics
        """
        try:
            health_status = {
                "healthy": True,
                "timestamp": torch.cuda.Event(enable_timing=True),
                "errors": [],
            }

            # Check CUDA availability
            if not torch.cuda.is_available():
                health_status["healthy"] = False
                health_status["errors"].append("CUDA not available")

            # Check GPU memory
            current_memory = await self._get_memory_info()
            memory_utilization = current_memory.get("utilization", 0)
            
            if memory_utilization > 95:
                health_status["healthy"] = False
                health_status["errors"].append(f"High memory utilization: {memory_utilization:.1f}%")

            # Test basic GPU operation
            try:
                test_tensor = torch.randn(100, 100, device=self.device)
                result = torch.matmul(test_tensor, test_tensor)
                del test_tensor, result
                torch.cuda.empty_cache()
            except Exception as e:
                health_status["healthy"] = False
                health_status["errors"].append(f"GPU operation test failed: {e}")

            health_status["memory_info"] = current_memory
            health_status["gpu_info"] = self.gpu_info

            return health_status

        except Exception as e:
            logger.error(f"GPU health check failed: {e}")
            return {
                "healthy": False,
                "errors": [f"Health check failed: {e}"],
                "timestamp": None,
            }

    def get_optimal_batch_size(self, model_memory_mb: int, input_size: Tuple[int, ...]) -> int:
        """
        Calculate optimal batch size based on available GPU memory.

        Args:
            model_memory_mb: Model memory usage in MB
            input_size: Input tensor size (H, W, C)

        Returns:
            Optimal batch size
        """
        try:
            # Get available memory in MB
            available_memory_mb = self.memory_info.get("free", 0) / (1024**2)
            
            # Estimate memory per sample (input + gradients + activations)
            h, w, c = input_size
            memory_per_sample_mb = (h * w * c * 4 * 3) / (1024**2)  # 4 bytes per float, 3x for gradients/activations
            
            # Calculate batch size with safety margin
            safety_margin = 0.8  # Use 80% of available memory
            usable_memory_mb = (available_memory_mb - model_memory_mb) * safety_margin
            
            batch_size = max(1, int(usable_memory_mb / memory_per_sample_mb))
            
            logger.info(f"Calculated optimal batch size: {batch_size}")
            return batch_size

        except Exception as e:
            logger.error(f"Error calculating batch size: {e}")
            return 1  # Fallback to batch size 1

    async def cleanup(self) -> None:
        """Clean up GPU resources."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logger.info("GPU cleanup completed")
        except Exception as e:
            logger.error(f"GPU cleanup error: {e}")


# Global GPU environment instance
gpu_env = GPUEnvironment()


async def get_gpu_environment() -> GPUEnvironment:
    """
    Get initialized GPU environment instance.

    Returns:
        GPUEnvironment instance
    """
    return gpu_env


async def initialize_gpu_environment() -> Dict[str, any]:
    """
    Initialize global GPU environment.

    Returns:
        GPU environment information
    """
    return await gpu_env.initialize()


async def gpu_health_check() -> Dict[str, any]:
    """
    Perform GPU health check.

    Returns:
        Health check results
    """
    return await gpu_env.health_check()


def get_device() -> torch.device:
    """
    Get current GPU device.

    Returns:
        PyTorch device
    """
    return gpu_env.device or torch.device("cpu")


def is_gpu_available() -> bool:
    """
    Check if GPU is available and initialized.

    Returns:
        True if GPU is available
    """
    return gpu_env.cuda_available and gpu_env.device is not None
