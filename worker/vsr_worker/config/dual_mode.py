"""
Configuration management for dual-mode worker architecture.

Provides environment-based configuration profiles for seamless switching
between CPU (local development) and GPU (production) processing modes.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from vsr_shared.logging import get_logger

logger = get_logger(__name__)


class WorkerMode(Enum):
    """Worker processing modes."""
    CPU = "cpu"
    GPU = "gpu"
    AUTO = "auto"


class ModelQuality(Enum):
    """Model quality levels."""
    LIGHTWEIGHT = "lightweight"
    STANDARD = "standard"
    HIGH = "high"


@dataclass
class ProcessingConfig:
    """Configuration for video processing parameters."""
    timeout_seconds: int = 300
    max_video_duration: int = 60
    max_resolution: tuple = (1920, 1080)
    batch_size: int = 1
    memory_limit_mb: Optional[int] = None
    thread_count: Optional[int] = None


@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    quality: ModelQuality = ModelQuality.STANDARD
    cache_dir: str = "/tmp/models"
    quantization: bool = False
    optimization_level: int = 1
    model_specific: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DualModeConfig:
    """Complete dual-mode worker configuration."""
    worker_mode: WorkerMode = WorkerMode.AUTO
    environment: str = "development"
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    debug: bool = False
    
    @classmethod
    def from_environment(cls) -> "DualModeConfig":
        """Create configuration from environment variables."""
        return cls(
            worker_mode=WorkerMode(os.getenv("WORKER_MODE", "auto").lower()),
            environment=os.getenv("ENVIRONMENT", "development").lower(),
            processing=ProcessingConfig(
                timeout_seconds=int(os.getenv("PROCESSING_TIMEOUT", "300")),
                max_video_duration=int(os.getenv("MAX_VIDEO_DURATION", "60")),
                max_resolution=tuple(map(int, os.getenv("MAX_RESOLUTION", "1920,1080").split(","))),
                batch_size=int(os.getenv("BATCH_SIZE", "1")),
                memory_limit_mb=int(os.getenv("MEMORY_LIMIT_MB")) if os.getenv("MEMORY_LIMIT_MB") else None,
                thread_count=int(os.getenv("CPU_THREADS")) if os.getenv("CPU_THREADS") else None,
            ),
            model=ModelConfig(
                quality=ModelQuality(os.getenv("MODEL_QUALITY", "standard").lower()),
                cache_dir=os.getenv("MODEL_CACHE_DIR", "/tmp/models"),
                quantization=os.getenv("MODEL_QUANTIZATION", "false").lower() == "true",
                optimization_level=int(os.getenv("MODEL_OPTIMIZATION", "1")),
            ),
            debug=os.getenv("DEBUG", "false").lower() == "true",
        )


class ConfigurationManager:
    """Manager for dual-mode worker configuration."""
    
    def __init__(self):
        """Initialize configuration manager."""
        self.config = DualModeConfig.from_environment()
        self._cpu_profiles = self._load_cpu_profiles()
        self._gpu_profiles = self._load_gpu_profiles()
        
    def get_config(self) -> DualModeConfig:
        """Get current configuration."""
        return self.config
    
    def get_processing_config(self, mode: WorkerMode) -> ProcessingConfig:
        """Get processing configuration for specific mode."""
        if mode == WorkerMode.CPU:
            return self._get_cpu_processing_config()
        elif mode == WorkerMode.GPU:
            return self._get_gpu_processing_config()
        else:
            return self.config.processing
    
    def get_model_config(self, mode: WorkerMode, model_name: str) -> Dict[str, Any]:
        """Get model configuration for specific mode and model."""
        base_config = self.config.model.model_specific.get(model_name, {})
        
        if mode == WorkerMode.CPU:
            profile = self._cpu_profiles.get(self.config.model.quality.value, {})
            return {**base_config, **profile.get(model_name, {})}
        elif mode == WorkerMode.GPU:
            profile = self._gpu_profiles.get(self.config.model.quality.value, {})
            return {**base_config, **profile.get(model_name, {})}
        else:
            return base_config
    
    def _get_cpu_processing_config(self) -> ProcessingConfig:
        """Get CPU-optimized processing configuration."""
        config = self.config.processing
        
        # CPU-specific adjustments
        return ProcessingConfig(
            timeout_seconds=max(config.timeout_seconds, 600),  # Longer timeout for CPU
            max_video_duration=min(config.max_video_duration, 30),  # Shorter videos for CPU
            max_resolution=(1280, 720),  # Lower resolution for CPU
            batch_size=1,  # Single frame processing for CPU
            memory_limit_mb=config.memory_limit_mb or 4096,  # 4GB default for CPU
            thread_count=config.thread_count or os.cpu_count(),
        )
    
    def _get_gpu_processing_config(self) -> ProcessingConfig:
        """Get GPU-optimized processing configuration."""
        config = self.config.processing
        
        # GPU-specific adjustments
        return ProcessingConfig(
            timeout_seconds=config.timeout_seconds,
            max_video_duration=config.max_video_duration,
            max_resolution=config.max_resolution,
            batch_size=max(config.batch_size, 4),  # Larger batches for GPU
            memory_limit_mb=config.memory_limit_mb or 8192,  # 8GB default for GPU
            thread_count=config.thread_count or 4,
        )
    
    def _load_cpu_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Load CPU model configuration profiles."""
        return {
            "lightweight": {
                "sttn": {
                    "num_frames": 3,
                    "img_size": (216, 120),
                    "patch_size": 12,
                    "embed_dim": 128,
                    "num_heads": 4,
                    "num_layers": 2,
                    "quantization": True,
                },
                "lama": {
                    "in_channels": 4,
                    "base_channels": 16,
                    "num_blocks": 2,
                    "use_fourier": False,
                    "quantization": True,
                },
                "propainter": {
                    "feature_channels": 64,
                    "num_layers": 2,
                    "use_temporal": False,
                    "fallback_only": True,
                }
            },
            "standard": {
                "sttn": {
                    "num_frames": 3,
                    "img_size": (432, 240),
                    "patch_size": 8,
                    "embed_dim": 256,
                    "num_heads": 4,
                    "num_layers": 3,
                    "quantization": True,
                },
                "lama": {
                    "in_channels": 4,
                    "base_channels": 32,
                    "num_blocks": 4,
                    "use_fourier": False,
                    "quantization": True,
                },
                "propainter": {
                    "feature_channels": 128,
                    "num_layers": 3,
                    "use_temporal": False,
                    "fallback_only": True,
                }
            },
            "high": {
                "sttn": {
                    "num_frames": 5,
                    "img_size": (432, 240),
                    "patch_size": 8,
                    "embed_dim": 384,
                    "num_heads": 6,
                    "num_layers": 4,
                    "quantization": False,
                },
                "lama": {
                    "in_channels": 4,
                    "base_channels": 48,
                    "num_blocks": 6,
                    "use_fourier": True,
                    "quantization": False,
                },
                "propainter": {
                    "feature_channels": 192,
                    "num_layers": 4,
                    "use_temporal": True,
                    "fallback_only": False,
                }
            }
        }
    
    def _load_gpu_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Load GPU model configuration profiles."""
        return {
            "lightweight": {
                "sttn": {
                    "num_frames": 5,
                    "img_size": (432, 240),
                    "patch_size": 8,
                    "embed_dim": 384,
                    "num_heads": 6,
                    "num_layers": 4,
                },
                "lama": {
                    "in_channels": 4,
                    "base_channels": 48,
                    "num_blocks": 6,
                    "use_fourier": True,
                },
                "propainter": {
                    "feature_channels": 192,
                    "num_layers": 4,
                    "use_temporal": True,
                }
            },
            "standard": {
                "sttn": {
                    "num_frames": 5,
                    "img_size": (432, 240),
                    "patch_size": 8,
                    "embed_dim": 512,
                    "num_heads": 8,
                    "num_layers": 6,
                },
                "lama": {
                    "in_channels": 4,
                    "base_channels": 64,
                    "num_blocks": 8,
                    "use_fourier": True,
                },
                "propainter": {
                    "feature_channels": 256,
                    "num_layers": 6,
                    "use_temporal": True,
                }
            },
            "high": {
                "sttn": {
                    "num_frames": 7,
                    "img_size": (864, 480),
                    "patch_size": 6,
                    "embed_dim": 768,
                    "num_heads": 12,
                    "num_layers": 8,
                },
                "lama": {
                    "in_channels": 4,
                    "base_channels": 96,
                    "num_blocks": 12,
                    "use_fourier": True,
                },
                "propainter": {
                    "feature_channels": 384,
                    "num_layers": 8,
                    "use_temporal": True,
                }
            }
        }
    
    def validate_config(self) -> bool:
        """Validate current configuration."""
        try:
            # Validate worker mode
            if self.config.worker_mode not in WorkerMode:
                logger.error(f"Invalid worker mode: {self.config.worker_mode}")
                return False
            
            # Validate model quality
            if self.config.model.quality not in ModelQuality:
                logger.error(f"Invalid model quality: {self.config.model.quality}")
                return False
            
            # Validate processing timeouts
            if self.config.processing.timeout_seconds <= 0:
                logger.error("Processing timeout must be positive")
                return False
            
            # Validate memory limits
            if self.config.processing.memory_limit_mb and self.config.processing.memory_limit_mb <= 0:
                logger.error("Memory limit must be positive")
                return False
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def get_environment_summary(self) -> Dict[str, Any]:
        """Get summary of current environment configuration."""
        return {
            "worker_mode": self.config.worker_mode.value,
            "environment": self.config.environment,
            "model_quality": self.config.model.quality.value,
            "processing_timeout": self.config.processing.timeout_seconds,
            "memory_limit_mb": self.config.processing.memory_limit_mb,
            "debug_enabled": self.config.debug,
            "model_cache_dir": self.config.model.cache_dir,
            "quantization_enabled": self.config.model.quantization,
        }
    
    def update_config(self, **kwargs) -> None:
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config {key} = {value}")
            else:
                logger.warning(f"Unknown config key: {key}")


# Global configuration instance
_config_manager = None


def get_config_manager() -> ConfigurationManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager


def get_dual_mode_config() -> DualModeConfig:
    """Get current dual-mode configuration."""
    return get_config_manager().get_config()


def get_processing_config(mode: WorkerMode) -> ProcessingConfig:
    """Get processing configuration for specific mode."""
    return get_config_manager().get_processing_config(mode)


def get_model_config(mode: WorkerMode, model_name: str) -> Dict[str, Any]:
    """Get model configuration for specific mode and model."""
    return get_config_manager().get_model_config(mode, model_name)


def validate_environment() -> bool:
    """Validate current environment configuration."""
    return get_config_manager().validate_config()


def print_config_summary():
    """Print configuration summary for debugging."""
    config_manager = get_config_manager()
    summary = config_manager.get_environment_summary()
    
    logger.info("=== Dual-Mode Worker Configuration ===")
    for key, value in summary.items():
        logger.info(f"{key}: {value}")
    logger.info("=====================================")


# Environment variable documentation
ENVIRONMENT_VARIABLES = {
    "WORKER_MODE": "Worker processing mode (cpu, gpu, auto) - default: auto",
    "ENVIRONMENT": "Deployment environment (development, production) - default: development",
    "MODEL_QUALITY": "Model quality level (lightweight, standard, high) - default: standard",
    "PROCESSING_TIMEOUT": "Processing timeout in seconds - default: 300",
    "MAX_VIDEO_DURATION": "Maximum video duration in seconds - default: 60",
    "MAX_RESOLUTION": "Maximum resolution as 'width,height' - default: 1920,1080",
    "BATCH_SIZE": "Processing batch size - default: 1",
    "MEMORY_LIMIT_MB": "Memory limit in MB - default: auto",
    "CPU_THREADS": "Number of CPU threads - default: auto",
    "MODEL_CACHE_DIR": "Model cache directory - default: /tmp/models",
    "MODEL_QUANTIZATION": "Enable model quantization (true, false) - default: false",
    "MODEL_OPTIMIZATION": "Model optimization level (0-3) - default: 1",
    "PREFER_GPU_LOCAL": "Prefer GPU in local development (true, false) - default: false",
    "DEBUG": "Enable debug mode (true, false) - default: false",
}
