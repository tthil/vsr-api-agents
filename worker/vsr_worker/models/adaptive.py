"""
Adaptive model interface for dual-mode CPU/GPU processing.

Provides unified model loading and processing interface that automatically
selects appropriate implementation based on worker environment.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from vsr_shared.logging import get_logger
from vsr_worker.gpu.environment import WorkerEnvironment

logger = get_logger(__name__)


class AdaptiveModelInterface(ABC):
    """Abstract interface for adaptive models supporting both CPU and GPU modes."""
    
    def __init__(self, model_name: str, worker_env: WorkerEnvironment):
        """
        Initialize adaptive model interface.
        
        Args:
            model_name: Name of the model (STTN, LAMA, ProPainter)
            worker_env: Worker environment instance
        """
        self.model_name = model_name
        self.worker_env = worker_env
        self.device = worker_env.device
        self.mode = worker_env.mode
        self.model = None
        self.loaded = False
        
    @abstractmethod
    async def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load model appropriate for current environment mode."""
        pass
    
    @abstractmethod
    async def inpaint_video(self, frames: np.ndarray, subtitle_area: Dict[str, int]) -> np.ndarray:
        """Process video frames to remove subtitles."""
        pass
    
    @abstractmethod
    async def unload_model(self) -> None:
        """Unload model and free memory."""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and status."""
        return {
            "model_name": self.model_name,
            "mode": self.mode,
            "device": str(self.device),
            "loaded": self.loaded,
            "memory_usage": self._get_memory_usage()
        }
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        if self.mode == "gpu" and torch.cuda.is_available():
            return {
                "gpu_memory_mb": torch.cuda.memory_allocated() / 1024**2,
                "gpu_memory_cached_mb": torch.cuda.memory_reserved() / 1024**2
            }
        else:
            import psutil
            process = psutil.Process()
            return {
                "cpu_memory_mb": process.memory_info().rss / 1024**2
            }


class AdaptiveModelFactory:
    """Factory for creating adaptive models based on environment."""
    
    def __init__(self, worker_env: WorkerEnvironment):
        """
        Initialize model factory.
        
        Args:
            worker_env: Worker environment instance
        """
        self.worker_env = worker_env
        self.loaded_models: Dict[str, AdaptiveModelInterface] = {}
    
    async def create_model(self, model_type: str) -> AdaptiveModelInterface:
        """
        Create adaptive model instance.
        
        Args:
            model_type: Type of model (sttn, lama, propainter)
            
        Returns:
            Adaptive model instance
        """
        model_type = model_type.lower()
        
        if model_type in self.loaded_models:
            return self.loaded_models[model_type]
        
        if model_type == "sttn":
            model = AdaptiveSTTNModel(self.worker_env)
        elif model_type == "lama":
            model = AdaptiveLAMAModel(self.worker_env)
        elif model_type == "propainter":
            model = AdaptiveProPainterModel(self.worker_env)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load the model
        await model.load_model()
        self.loaded_models[model_type] = model
        
        logger.info(f"Created {model_type} model in {self.worker_env.mode} mode")
        return model
    
    async def unload_all_models(self):
        """Unload all models and free memory."""
        for model_name, model in self.loaded_models.items():
            await model.unload_model()
            logger.info(f"Unloaded {model_name} model")
        
        self.loaded_models.clear()
        
        # Clear GPU cache if in GPU mode
        if self.worker_env.mode == "gpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()


class AdaptiveSTTNModel(AdaptiveModelInterface):
    """Adaptive STTN model supporting both CPU and GPU modes."""
    
    def __init__(self, worker_env: WorkerEnvironment):
        super().__init__("STTN", worker_env)
        self.config = self._get_model_config()
    
    def _get_model_config(self) -> Dict[str, Any]:
        """Get model configuration based on mode."""
        if self.mode == "gpu":
            return {
                "num_frames": 5,
                "img_size": (432, 240),
                "patch_size": 8,
                "embed_dim": 512,
                "num_heads": 8,
                "num_layers": 6
            }
        else:  # CPU mode - lighter configuration
            return {
                "num_frames": 3,
                "img_size": (216, 120),
                "patch_size": 12,
                "embed_dim": 256,
                "num_heads": 4,
                "num_layers": 3
            }
    
    async def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load STTN model appropriate for current mode."""
        try:
            if self.mode == "gpu":
                # Load full GPU model
                from vsr_worker.models.sttn import STTNModel
                self.model = STTNModel(self.config).to(self.device)
            else:
                # Load CPU-optimized model
                self.model = CPUSTTNModel(self.config)
            
            self.loaded = True
            logger.info(f"STTN model loaded in {self.mode} mode")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load STTN model: {e}")
            return False
    
    async def inpaint_video(self, frames: np.ndarray, subtitle_area: Dict[str, int]) -> np.ndarray:
        """Process video frames using STTN model."""
        if not self.loaded:
            raise RuntimeError("Model not loaded")
        
        if self.mode == "gpu":
            return await self._process_gpu(frames, subtitle_area)
        else:
            return await self._process_cpu(frames, subtitle_area)
    
    async def _process_gpu(self, frames: np.ndarray, subtitle_area: Dict[str, int]) -> np.ndarray:
        """Process frames using GPU model."""
        # Convert frames to tensor
        frames_tensor = torch.from_numpy(frames).float().to(self.device)
        frames_tensor = frames_tensor.permute(0, 3, 1, 2) / 255.0  # [T, C, H, W]
        
        # Create mask for subtitle area
        mask = self._create_mask(frames_tensor.shape, subtitle_area)
        
        # Process with model
        with torch.no_grad():
            output = self.model(frames_tensor.unsqueeze(0), mask.unsqueeze(0))
            output = output.squeeze(0)
        
        # Convert back to numpy
        output = output.permute(0, 2, 3, 1).cpu().numpy()
        output = np.clip(output * 255, 0, 255).astype(np.uint8)
        
        return output
    
    async def _process_cpu(self, frames: np.ndarray, subtitle_area: Dict[str, int]) -> np.ndarray:
        """Process frames using CPU model."""
        # Resize frames for CPU processing
        import cv2
        
        processed_frames = []
        h, w = self.config["img_size"]
        
        for frame in frames:
            # Resize frame
            resized = cv2.resize(frame, (w, h))
            
            # Simple inpainting for subtitle area
            mask = np.zeros((h, w), dtype=np.uint8)
            y1 = int(subtitle_area["y"] * h / frame.shape[0])
            y2 = int((subtitle_area["y"] + subtitle_area["height"]) * h / frame.shape[0])
            x1 = int(subtitle_area["x"] * w / frame.shape[1])
            x2 = int((subtitle_area["x"] + subtitle_area["width"]) * w / frame.shape[1])
            
            mask[y1:y2, x1:x2] = 255
            
            # Use OpenCV inpainting
            inpainted = cv2.inpaint(resized, mask, 3, cv2.INPAINT_TELEA)
            
            # Resize back to original size
            result = cv2.resize(inpainted, (frame.shape[1], frame.shape[0]))
            processed_frames.append(result)
        
        return np.array(processed_frames)
    
    def _create_mask(self, shape: tuple, subtitle_area: Dict[str, int]) -> torch.Tensor:
        """Create mask tensor for subtitle area."""
        T, C, H, W = shape
        mask = torch.zeros((T, 1, H, W), device=self.device)
        
        # Calculate mask coordinates
        y1 = int(subtitle_area["y"] * H / 100)
        y2 = int((subtitle_area["y"] + subtitle_area["height"]) * H / 100)
        x1 = int(subtitle_area["x"] * W / 100)
        x2 = int((subtitle_area["x"] + subtitle_area["width"]) * W / 100)
        
        mask[:, :, y1:y2, x1:x2] = 1.0
        return mask
    
    async def unload_model(self) -> None:
        """Unload STTN model."""
        if self.model is not None:
            del self.model
            self.model = None
        
        self.loaded = False
        
        if self.mode == "gpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()


class AdaptiveLAMAModel(AdaptiveModelInterface):
    """Adaptive LAMA model supporting both CPU and GPU modes."""
    
    def __init__(self, worker_env: WorkerEnvironment):
        super().__init__("LAMA", worker_env)
        self.config = self._get_model_config()
    
    def _get_model_config(self) -> Dict[str, Any]:
        """Get model configuration based on mode."""
        if self.mode == "gpu":
            return {
                "in_channels": 4,
                "base_channels": 64,
                "num_blocks": 8,
                "use_fourier": True
            }
        else:  # CPU mode
            return {
                "in_channels": 4,
                "base_channels": 32,
                "num_blocks": 4,
                "use_fourier": False
            }
    
    async def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load LAMA model appropriate for current mode."""
        try:
            if self.mode == "gpu":
                from vsr_worker.models.lama import LAMAModel
                self.model = LAMAModel(self.config).to(self.device)
            else:
                self.model = CPULAMAModel(self.config)
            
            self.loaded = True
            logger.info(f"LAMA model loaded in {self.mode} mode")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load LAMA model: {e}")
            return False
    
    async def inpaint_video(self, frames: np.ndarray, subtitle_area: Dict[str, int]) -> np.ndarray:
        """Process video frames using LAMA model."""
        if not self.loaded:
            raise RuntimeError("Model not loaded")
        
        # LAMA processes frame by frame
        processed_frames = []
        
        for frame in frames:
            if self.mode == "gpu":
                processed_frame = await self._process_frame_gpu(frame, subtitle_area)
            else:
                processed_frame = await self._process_frame_cpu(frame, subtitle_area)
            
            processed_frames.append(processed_frame)
        
        return np.array(processed_frames)
    
    async def _process_frame_gpu(self, frame: np.ndarray, subtitle_area: Dict[str, int]) -> np.ndarray:
        """Process single frame using GPU LAMA model."""
        import cv2
        
        # Create mask
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        y1, y2 = subtitle_area["y"], subtitle_area["y"] + subtitle_area["height"]
        x1, x2 = subtitle_area["x"], subtitle_area["x"] + subtitle_area["width"]
        mask[y1:y2, x1:x2] = 255
        
        # Convert to tensors
        frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).to(self.device) / 255.0
        
        # Concatenate frame and mask
        input_tensor = torch.cat([frame_tensor, mask_tensor], dim=1)
        
        # Process with model
        with torch.no_grad():
            output = self.model(input_tensor)
            output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        output = np.clip(output * 255, 0, 255).astype(np.uint8)
        return output
    
    async def _process_frame_cpu(self, frame: np.ndarray, subtitle_area: Dict[str, int]) -> np.ndarray:
        """Process single frame using CPU fallback."""
        import cv2
        
        # Create mask
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        y1, y2 = subtitle_area["y"], subtitle_area["y"] + subtitle_area["height"]
        x1, x2 = subtitle_area["x"], subtitle_area["x"] + subtitle_area["width"]
        mask[y1:y2, x1:x2] = 255
        
        # Use OpenCV inpainting
        result = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
        return result
    
    async def unload_model(self) -> None:
        """Unload LAMA model."""
        if self.model is not None:
            del self.model
            self.model = None
        
        self.loaded = False
        
        if self.mode == "gpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()


class AdaptiveProPainterModel(AdaptiveModelInterface):
    """Adaptive ProPainter model supporting both CPU and GPU modes."""
    
    def __init__(self, worker_env: WorkerEnvironment):
        super().__init__("ProPainter", worker_env)
        self.config = self._get_model_config()
    
    def _get_model_config(self) -> Dict[str, Any]:
        """Get model configuration based on mode."""
        if self.mode == "gpu":
            return {
                "feature_channels": 256,
                "num_layers": 6,
                "use_temporal": True
            }
        else:  # CPU mode
            return {
                "feature_channels": 128,
                "num_layers": 3,
                "use_temporal": False
            }
    
    async def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load ProPainter model appropriate for current mode."""
        try:
            if self.mode == "gpu":
                from vsr_worker.models.propainter import ProPainterModel
                self.model = ProPainterModel(self.config).to(self.device)
            else:
                # CPU mode uses simple OpenCV inpainting
                self.model = "cpu_fallback"
            
            self.loaded = True
            logger.info(f"ProPainter model loaded in {self.mode} mode")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ProPainter model: {e}")
            return False
    
    async def inpaint_video(self, frames: np.ndarray, subtitle_area: Dict[str, int]) -> np.ndarray:
        """Process video frames using ProPainter model."""
        if not self.loaded:
            raise RuntimeError("Model not loaded")
        
        if self.mode == "gpu":
            return await self._process_gpu(frames, subtitle_area)
        else:
            return await self._process_cpu(frames, subtitle_area)
    
    async def _process_gpu(self, frames: np.ndarray, subtitle_area: Dict[str, int]) -> np.ndarray:
        """Process frames using GPU ProPainter model."""
        # Convert to tensors
        frames_tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2).to(self.device) / 255.0
        
        # Create mask
        mask = self._create_mask(frames_tensor.shape, subtitle_area)
        
        # Process with model
        with torch.no_grad():
            output = self.model(frames_tensor.unsqueeze(0), mask.unsqueeze(0))
            output = output.squeeze(0)
        
        # Convert back to numpy
        output = output.permute(0, 2, 3, 1).cpu().numpy()
        output = np.clip(output * 255, 0, 255).astype(np.uint8)
        
        return output
    
    async def _process_cpu(self, frames: np.ndarray, subtitle_area: Dict[str, int]) -> np.ndarray:
        """Process frames using CPU fallback."""
        import cv2
        
        processed_frames = []
        
        for frame in frames:
            # Create mask
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            y1, y2 = subtitle_area["y"], subtitle_area["y"] + subtitle_area["height"]
            x1, x2 = subtitle_area["x"], subtitle_area["x"] + subtitle_area["width"]
            mask[y1:y2, x1:x2] = 255
            
            # Use OpenCV inpainting
            result = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
            processed_frames.append(result)
        
        return np.array(processed_frames)
    
    def _create_mask(self, shape: tuple, subtitle_area: Dict[str, int]) -> torch.Tensor:
        """Create mask tensor for subtitle area."""
        T, C, H, W = shape
        mask = torch.zeros((T, 1, H, W), device=self.device)
        
        # Calculate mask coordinates
        y1 = int(subtitle_area["y"] * H / 100)
        y2 = int((subtitle_area["y"] + subtitle_area["height"]) * H / 100)
        x1 = int(subtitle_area["x"] * W / 100)
        x2 = int((subtitle_area["x"] + subtitle_area["width"]) * W / 100)
        
        mask[:, :, y1:y2, x1:x2] = 1.0
        return mask
    
    async def unload_model(self) -> None:
        """Unload ProPainter model."""
        if self.model is not None and self.model != "cpu_fallback":
            del self.model
            self.model = None
        
        self.loaded = False
        
        if self.mode == "gpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()


# CPU-optimized model implementations
class CPUSTTNModel(nn.Module):
    """Lightweight STTN model optimized for CPU processing."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        # Simplified architecture for CPU
        # This would be implemented with reduced complexity
        
    def forward(self, x, mask):
        # Simplified forward pass
        return x  # Placeholder


class CPULAMAModel(nn.Module):
    """Lightweight LAMA model optimized for CPU processing."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        # Simplified architecture for CPU
        
    def forward(self, x):
        # Simplified forward pass
        return x[:, :3]  # Return only RGB channels
