"""
LAMA (Large Mask Inpainting) model implementation for video inpainting.
Optimized for H100/A100 GPUs with efficient memory management.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any
import cv2
from pathlib import Path

from vsr_worker.gpu.environment import GPUEnvironment
from vsr_worker.gpu.model_loader import BaseModelLoader


class ResNetBlock(nn.Module):
    """ResNet block with dilated convolutions."""
    
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        
        # Skip connection
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x):
        residual = self.skip(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        out = out + residual
        out = self.activation(out)
        
        return out


class FourierFeatureTransform(nn.Module):
    """Fourier Feature Transform for better texture synthesis."""
    
    def __init__(self, num_input_channels: int, mapping_size: int = 256, scale: float = 10.0):
        super().__init__()
        self.num_input_channels = num_input_channels
        self.mapping_size = mapping_size
        self.register_buffer('B', torch.randn(mapping_size, num_input_channels) * scale)
        
    def forward(self, x):
        # x shape: [B, C, H, W]
        B, C, H, W = x.shape
        
        # Flatten spatial dimensions
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        
        # Apply Fourier features
        x_proj = torch.matmul(x_flat, self.B.T)  # [B*H*W, mapping_size]
        fourier_features = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)  # [B*H*W, 2*mapping_size]
        
        # Reshape back
        fourier_features = fourier_features.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # [B, 2*mapping_size, H, W]
        
        return fourier_features


class LAMAModel(nn.Module):
    """LAMA model for image inpainting with Fourier features and dilated convolutions."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Model parameters
        self.in_channels = config.get("in_channels", 4)  # RGB + mask
        self.base_channels = config.get("base_channels", 64)
        self.num_blocks = config.get("num_blocks", 8)
        self.use_fourier = config.get("use_fourier", True)
        
        # Fourier feature transform
        if self.use_fourier:
            self.fourier_transform = FourierFeatureTransform(
                num_input_channels=2,  # x, y coordinates
                mapping_size=128,
                scale=10.0
            )
            fourier_channels = 256  # 2 * mapping_size
        else:
            fourier_channels = 0
        
        # Input projection
        input_channels = self.in_channels + fourier_channels
        self.input_conv = nn.Conv2d(input_channels, self.base_channels, 3, padding=1)
        
        # Encoder blocks with increasing dilation
        self.encoder_blocks = nn.ModuleList()
        dilations = [1, 2, 4, 8, 16, 8, 4, 2]
        
        for i in range(self.num_blocks):
            dilation = dilations[i % len(dilations)]
            self.encoder_blocks.append(
                ResNetBlock(self.base_channels, self.base_channels, dilation=dilation)
            )
        
        # Output projection
        self.output_conv = nn.Sequential(
            nn.Conv2d(self.base_channels, self.base_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.base_channels // 2, 3, 3, padding=1),
            nn.Tanh()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _create_coordinate_grid(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Create normalized coordinate grid."""
        y_coords = torch.linspace(-1, 1, height, device=device)
        x_coords = torch.linspace(-1, 1, width, device=device)
        
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        coord_grid = torch.stack([x_grid, y_grid], dim=0)  # [2, H, W]
        
        return coord_grid
    
    def forward(self, masked_image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for LAMA model.
        
        Args:
            masked_image: Input image with masked regions [B, 3, H, W]
            mask: Binary mask [B, 1, H, W] (1 = inpaint, 0 = keep)
            
        Returns:
            Inpainted image [B, 3, H, W]
        """
        B, _, H, W = masked_image.shape
        device = masked_image.device
        
        # Concatenate image and mask
        input_tensor = torch.cat([masked_image, mask], dim=1)  # [B, 4, H, W]
        
        # Add Fourier features if enabled
        if self.use_fourier:
            # Create coordinate grid
            coord_grid = self._create_coordinate_grid(H, W, device)
            coord_grid = coord_grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, 2, H, W]
            
            # Apply Fourier transform
            fourier_features = self.fourier_transform(coord_grid)  # [B, 256, H, W]
            
            # Concatenate with input
            input_tensor = torch.cat([input_tensor, fourier_features], dim=1)
        
        # Input projection
        features = self.input_conv(input_tensor)
        
        # Apply encoder blocks
        for block in self.encoder_blocks:
            features = block(features)
        
        # Output projection
        output = self.output_conv(features)
        
        # Blend with original image
        output = (output + 1) / 2  # Convert from [-1, 1] to [0, 1]
        result = masked_image * (1 - mask) + output * mask
        
        return result


class LAMAModelLoader(BaseModelLoader):
    """Model loader for LAMA model."""
    
    def __init__(self, gpu_env: GPUEnvironment):
        super().__init__(gpu_env)
        self.model_name = "LAMA"
        self.model_config = {
            "in_channels": 4,
            "base_channels": 64,
            "num_blocks": 8,
            "use_fourier": True
        }
    
    async def load_model(self, model_path: Optional[str] = None) -> LAMAModel:
        """Load LAMA model with optimizations."""
        try:
            # Check GPU memory
            if not self.gpu_env.check_memory_available(1536):  # 1.5GB minimum
                raise RuntimeError("Insufficient GPU memory for LAMA model")
            
            # Create model
            model = LAMAModel(self.model_config)
            model = model.to(self.gpu_env.device)
            
            # Load pretrained weights if available
            if model_path and Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location=self.gpu_env.device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print(f"Loaded LAMA model from {model_path}")
            else:
                print("Using randomly initialized LAMA model (no pretrained weights)")
            
            # Optimize for inference
            model.eval()
            
            # Enable optimizations
            if hasattr(torch, 'compile') and self.gpu_env.device.type == 'cuda':
                try:
                    model = torch.compile(model, mode='reduce-overhead')
                    print("Enabled torch.compile optimization for LAMA")
                except Exception as e:
                    print(f"Failed to compile LAMA model: {e}")
            
            # Enable mixed precision if supported
            if self.gpu_env.device.type == 'cuda':
                torch.backends.cudnn.benchmark = True
                torch.set_float32_matmul_precision('high')
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load LAMA model: {e}")
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess single frame for LAMA model.
        
        Args:
            frame: Input frame [H, W, C] in uint8 format
            
        Returns:
            Preprocessed frame [1, C, H, W] in float32 format
        """
        # Convert to float and normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        
        return frame_tensor.to(self.gpu_env.device)
    
    def create_subtitle_mask(self, frame_shape: Tuple[int, int, int], subtitle_area: Dict[str, int]) -> torch.Tensor:
        """
        Create binary mask for subtitle area.
        
        Args:
            frame_shape: Shape of input frame [H, W, C]
            subtitle_area: Dictionary with x1, y1, x2, y2 coordinates
            
        Returns:
            Binary mask [1, 1, H, W]
        """
        H, W, C = frame_shape
        
        # Create mask
        mask = np.zeros((H, W), dtype=np.float32)
        
        # Fill subtitle area
        y1, y2 = subtitle_area['y1'], subtitle_area['y2']
        x1, x2 = subtitle_area['x1'], subtitle_area['x2']
        
        # Ensure coordinates are within bounds
        y1, y2 = max(0, y1), min(H, y2)
        x1, x2 = max(0, x1), min(W, x2)
        
        mask[y1:y2, x1:x2] = 1.0
        
        # Convert to tensor
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        return mask_tensor.to(self.gpu_env.device)
    
    def postprocess_frame(self, frame: torch.Tensor) -> np.ndarray:
        """
        Postprocess output frame from LAMA model.
        
        Args:
            frame: Output frame [1, C, H, W] in float32 format
            
        Returns:
            Postprocessed frame [H, W, C] in uint8 format
        """
        # Remove batch dimension and rearrange
        frame = frame.squeeze(0).permute(1, 2, 0)  # [H, W, C]
        
        # Clamp to [0, 1] and convert to uint8
        frame = torch.clamp(frame, 0, 1)
        frame = (frame * 255).cpu().numpy().astype(np.uint8)
        
        return frame
    
    async def inpaint_video(self, frames: np.ndarray, subtitle_area: Dict[str, int]) -> np.ndarray:
        """
        Perform video inpainting using LAMA model (frame by frame).
        
        Args:
            frames: Input video frames [T, H, W, C]
            subtitle_area: Subtitle area coordinates
            
        Returns:
            Inpainted video frames [T, H, W, C]
        """
        if not hasattr(self, '_model') or self._model is None:
            raise RuntimeError("LAMA model not loaded")
        
        try:
            T, H, W, C = frames.shape
            output_frames = np.zeros_like(frames)
            
            # Create mask once (same for all frames)
            mask_tensor = self.create_subtitle_mask((H, W, C), subtitle_area)
            
            with torch.no_grad():
                for i in range(T):
                    # Preprocess frame
                    frame_tensor = self.preprocess_frame(frames[i])
                    
                    # Apply mask to frame
                    masked_frame = frame_tensor * (1 - mask_tensor)
                    
                    # Inpaint frame
                    with torch.cuda.amp.autocast(enabled=self.gpu_env.device.type == 'cuda'):
                        inpainted_frame = self._model(masked_frame, mask_tensor)
                    
                    # Postprocess
                    output_frames[i] = self.postprocess_frame(inpainted_frame)
                
                return output_frames
                
        except Exception as e:
            raise RuntimeError(f"LAMA inpainting failed: {e}")
    
    def estimate_processing_time(self, num_frames: int, resolution: Tuple[int, int]) -> float:
        """
        Estimate processing time for LAMA model.
        
        Args:
            num_frames: Number of frames to process
            resolution: Frame resolution (width, height)
            
        Returns:
            Estimated processing time in seconds
        """
        # Base time per frame (empirically determined)
        base_time_per_frame = 0.1  # seconds
        
        # Resolution factor (normalized to 1080p)
        width, height = resolution
        resolution_factor = (width * height) / (1920 * 1080)
        
        # GPU factor (H100 is faster than A100)
        gpu_factor = 0.7 if "H100" in str(self.gpu_env.device) else 1.0
        
        total_time = num_frames * base_time_per_frame * resolution_factor * gpu_factor
        
        return max(total_time, 1.0)  # Minimum 1 second
