"""Model loading infrastructure for VSR worker."""

import os
import hashlib
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, snapshot_download

from vsr_shared.logging import get_logger
from .environment import get_gpu_environment, get_device

logger = get_logger(__name__)


class ModelLoader(ABC):
    """Abstract base class for model loaders."""

    def __init__(self, model_name: str, cache_dir: Optional[str] = None):
        """
        Initialize model loader.

        Args:
            model_name: Name of the model
            cache_dir: Directory to cache models
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir or os.getenv("MODEL_CACHE_DIR", "/tmp/models"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.device = None
        self.loaded = False

    @abstractmethod
    async def load_model(self) -> nn.Module:
        """Load and return the model."""
        pass

    @abstractmethod
    async def download_model(self) -> str:
        """Download model files and return path."""
        pass

    async def initialize(self) -> nn.Module:
        """
        Initialize model with GPU environment.

        Returns:
            Loaded model instance
        """
        try:
            # Get GPU environment
            gpu_env = await get_gpu_environment()
            self.device = get_device()

            logger.info(f"Initializing {self.model_name} model on {self.device}")

            # Download model if needed
            model_path = await self.download_model()
            
            # Load model
            self.model = await self.load_model()
            
            # Move to GPU
            if self.device.type == "cuda":
                self.model = self.model.to(self.device)
                
                # Enable mixed precision if supported
                if gpu_env.gpu_info.get("optimization_level") in ["h100", "a100"]:
                    self.model = self.model.half()  # Use FP16
                    logger.info("Enabled FP16 precision for high-end GPU")

            self.loaded = True
            logger.info(f"{self.model_name} model loaded successfully")
            
            return self.model

        except Exception as e:
            logger.error(f"Failed to initialize {self.model_name} model: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if not self.loaded:
            return {"loaded": False}

        try:
            model_info = {
                "loaded": True,
                "model_name": self.model_name,
                "device": str(self.device),
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            }

            # Get memory usage if on GPU
            if self.device.type == "cuda":
                model_info["gpu_memory_mb"] = torch.cuda.memory_allocated(self.device) / (1024**2)

            return model_info

        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"loaded": False, "error": str(e)}

    async def unload_model(self) -> None:
        """Unload model and free memory."""
        try:
            if self.model is not None:
                del self.model
                self.model = None
                
            if self.device and self.device.type == "cuda":
                torch.cuda.empty_cache()
                
            self.loaded = False
            logger.info(f"{self.model_name} model unloaded")

        except Exception as e:
            logger.error(f"Error unloading {self.model_name} model: {e}")


class HuggingFaceModelLoader(ModelLoader):
    """Model loader for Hugging Face models."""

    def __init__(
        self,
        model_name: str,
        repo_id: str,
        model_file: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Hugging Face model loader.

        Args:
            model_name: Name of the model
            repo_id: Hugging Face repository ID
            model_file: Specific model file to download
            cache_dir: Cache directory
            **kwargs: Additional arguments
        """
        super().__init__(model_name, cache_dir)
        self.repo_id = repo_id
        self.model_file = model_file
        self.kwargs = kwargs

    async def download_model(self) -> str:
        """Download model from Hugging Face Hub."""
        try:
            logger.info(f"Downloading {self.model_name} from {self.repo_id}")

            if self.model_file:
                # Download specific file
                model_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=self.model_file,
                    cache_dir=str(self.cache_dir),
                )
            else:
                # Download entire repository
                model_path = snapshot_download(
                    repo_id=self.repo_id,
                    cache_dir=str(self.cache_dir),
                )

            logger.info(f"Model downloaded to: {model_path}")
            return model_path

        except Exception as e:
            logger.error(f"Failed to download {self.model_name}: {e}")
            raise

    async def load_model(self) -> nn.Module:
        """Load model from downloaded files."""
        # This will be implemented by specific model loaders
        raise NotImplementedError("Subclasses must implement load_model")


class LocalModelLoader(ModelLoader):
    """Model loader for local model files."""

    def __init__(
        self,
        model_name: str,
        model_path: str,
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize local model loader.

        Args:
            model_name: Name of the model
            model_path: Path to local model file
            cache_dir: Cache directory
            **kwargs: Additional arguments
        """
        super().__init__(model_name, cache_dir)
        self.model_path = Path(model_path)
        self.kwargs = kwargs

    async def download_model(self) -> str:
        """Return local model path (no download needed)."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        return str(self.model_path)

    async def load_model(self) -> nn.Module:
        """Load model from local file."""
        # This will be implemented by specific model loaders
        raise NotImplementedError("Subclasses must implement load_model")


class ModelManager:
    """Manager for multiple models with caching and memory optimization."""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize model manager.

        Args:
            cache_dir: Directory to cache models
        """
        self.cache_dir = cache_dir
        self.models: Dict[str, ModelLoader] = {}
        self.loaded_models: Dict[str, nn.Module] = {}
        self.model_configs: Dict[str, Dict[str, Any]] = {}

    def register_model(
        self,
        model_name: str,
        loader_class: type,
        **loader_kwargs
    ) -> None:
        """
        Register a model with the manager.

        Args:
            model_name: Name of the model
            loader_class: ModelLoader subclass
            **loader_kwargs: Arguments for the loader
        """
        self.model_configs[model_name] = {
            "loader_class": loader_class,
            "loader_kwargs": loader_kwargs,
        }
        logger.info(f"Registered model: {model_name}")

    async def load_model(self, model_name: str) -> nn.Module:
        """
        Load a model by name.

        Args:
            model_name: Name of the model to load

        Returns:
            Loaded model instance
        """
        if model_name in self.loaded_models:
            logger.info(f"Model {model_name} already loaded")
            return self.loaded_models[model_name]

        if model_name not in self.model_configs:
            raise ValueError(f"Model {model_name} not registered")

        try:
            # Create loader
            config = self.model_configs[model_name]
            loader_class = config["loader_class"]
            loader_kwargs = config["loader_kwargs"]
            
            if self.cache_dir:
                loader_kwargs["cache_dir"] = self.cache_dir

            loader = loader_class(model_name=model_name, **loader_kwargs)
            self.models[model_name] = loader

            # Load model
            model = await loader.initialize()
            self.loaded_models[model_name] = model

            logger.info(f"Model {model_name} loaded successfully")
            return model

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    async def unload_model(self, model_name: str) -> None:
        """
        Unload a model to free memory.

        Args:
            model_name: Name of the model to unload
        """
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]

        if model_name in self.models:
            await self.models[model_name].unload_model()
            del self.models[model_name]

        logger.info(f"Model {model_name} unloaded")

    async def unload_all_models(self) -> None:
        """Unload all models to free memory."""
        model_names = list(self.loaded_models.keys())
        for model_name in model_names:
            await self.unload_model(model_name)
        
        logger.info("All models unloaded")

    def get_loaded_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about loaded models."""
        info = {}
        for model_name, loader in self.models.items():
            if model_name in self.loaded_models:
                info[model_name] = loader.get_model_info()
        return info

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all loaded models."""
        health_status = {
            "healthy": True,
            "models": {},
            "total_models": len(self.loaded_models),
            "total_memory_mb": 0,
        }

        for model_name, loader in self.models.items():
            if model_name in self.loaded_models:
                model_info = loader.get_model_info()
                health_status["models"][model_name] = model_info
                
                if "gpu_memory_mb" in model_info:
                    health_status["total_memory_mb"] += model_info["gpu_memory_mb"]

        return health_status


# Global model manager instance
model_manager = ModelManager()


async def get_model_manager() -> ModelManager:
    """
    Get global model manager instance.

    Returns:
        ModelManager instance
    """
    return model_manager


async def load_model(model_name: str) -> nn.Module:
    """
    Load a model by name using global manager.

    Args:
        model_name: Name of the model to load

    Returns:
        Loaded model instance
    """
    return await model_manager.load_model(model_name)


async def unload_model(model_name: str) -> None:
    """
    Unload a model by name using global manager.

    Args:
        model_name: Name of the model to unload
    """
    await model_manager.unload_model(model_name)
