"""GPU package for VSR worker."""

from .environment import (
    GPUEnvironment,
    gpu_env,
    get_gpu_environment,
    initialize_gpu_environment,
    gpu_health_check,
    get_device,
    is_gpu_available,
)

from .model_loader import (
    ModelLoader,
    HuggingFaceModelLoader,
    LocalModelLoader,
    ModelManager,
    model_manager,
    get_model_manager,
    load_model,
    unload_model,
)

__all__ = [
    "GPUEnvironment",
    "gpu_env",
    "get_gpu_environment",
    "initialize_gpu_environment",
    "gpu_health_check",
    "get_device",
    "is_gpu_available",
    "ModelLoader",
    "HuggingFaceModelLoader",
    "LocalModelLoader",
    "ModelManager",
    "model_manager",
    "get_model_manager",
    "load_model",
    "unload_model",
]
