"""
STTN (Spatial-Temporal Transformer Network) model implementation for video inpainting.
Optimized for H100/A100 GPUs with efficient memory management.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, Any
import cv2
from pathlib import Path

from vsr_worker.gpu.environment import GPUEnvironment
from vsr_worker.gpu.model_loader import BaseModelLoader


class STTNModel(nn.Module):
    """STTN model for video inpainting with subtitle removal."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model architecture parameters
        self.num_frames = config.get("num_frames", 5)
        self.img_size = config.get("img_size", (432, 240))  # Optimized for memory
        self.patch_size = config.get("patch_size", 8)
        self.embed_dim = config.get("embed_dim", 512)
        self.num_heads = config.get("num_heads", 8)
        self.num_layers = config.get("num_layers", 6)
        
        # Build model components
        self._build_model()
        
    def _build_model(self):
        """Build STTN model architecture."""
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, self.embed_dim, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
        
        # Positional encoding
        h_patches = self.img_size[1] // self.patch_size
        w_patches = self.img_size[0] // self.patch_size
        num_patches = h_patches * w_patches
        
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, self.embed_dim)
        )
        self.temporal_embed = nn.Parameter(
            torch.zeros(1, self.num_frames, self.embed_dim)
        )
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.num_layers
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.GELU(),
            nn.Linear(self.embed_dim * 2, 3 * self.patch_size * self.patch_size)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, frames: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for STTN model.
        
        Args:
            frames: Input video frames [B, T, C, H, W]
            masks: Binary masks for inpainting [B, T, 1, H, W]
            
        Returns:
            Inpainted frames [B, T, C, H, W]
        """
        B, T, C, H, W = frames.shape
        
        # Resize frames to model input size
        frames_resized = torch.nn.functional.interpolate(
            frames.view(B * T, C, H, W),
            size=self.img_size,
            mode='bilinear',
            align_corners=False
        ).view(B, T, C, self.img_size[1], self.img_size[0])
        
        masks_resized = torch.nn.functional.interpolate(
            masks.view(B * T, 1, H, W),
            size=self.img_size,
            mode='nearest'
        ).view(B, T, 1, self.img_size[1], self.img_size[0])
        
        # Apply masks to frames
        masked_frames = frames_resized * (1 - masks_resized)
        
        # Patch embedding
        patches = self.patch_embed(masked_frames.view(B * T, C, self.img_size[1], self.img_size[0]))
        patches = patches.flatten(2).transpose(1, 2)  # [B*T, N, D]
        patches = patches.view(B, T, -1, self.embed_dim)  # [B, T, N, D]
        
        # Add positional embeddings
        patches = patches + self.pos_embed.unsqueeze(1)
        patches = patches + self.temporal_embed.unsqueeze(2)
        
        # Reshape for transformer
        patches = patches.view(B, T * patches.size(2), self.embed_dim)
        
        # Apply transformer
        features = self.transformer(patches)
        
        # Output projection
        output = self.output_proj(features)
        
        # Reshape to image patches
        N = patches.size(1) // T
        output = output.view(B, T, N, 3, self.patch_size, self.patch_size)
        
        # Reconstruct images
        h_patches = self.img_size[1] // self.patch_size
        w_patches = self.img_size[0] // self.patch_size
        
        reconstructed = output.view(B, T, h_patches, w_patches, 3, self.patch_size, self.patch_size)
        reconstructed = reconstructed.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        reconstructed = reconstructed.view(B, T, 3, self.img_size[1], self.img_size[0])
        
        # Resize back to original size
        reconstructed = torch.nn.functional.interpolate(
            reconstructed.view(B * T, 3, self.img_size[1], self.img_size[0]),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).view(B, T, 3, H, W)
        
        # Blend with original frames
        output_frames = frames * (1 - masks) + reconstructed * masks
        
        return output_frames


class STTNModelLoader(BaseModelLoader):
    """Model loader for STTN model."""
    
    def __init__(self, gpu_env: GPUEnvironment):
        super().__init__(gpu_env)
        self.model_name = "STTN"
        self.model_config = {
            "num_frames": 5,
            "img_size": (432, 240),
            "patch_size": 8,
            "embed_dim": 512,
            "num_heads": 8,
            "num_layers": 6
        }
    
    async def load_model(self, model_path: Optional[str] = None) -> STTNModel:
        """Load STTN model with optimizations."""
        try:
            # Check GPU memory
            if not self.gpu_env.check_memory_available(2048):  # 2GB minimum
                raise RuntimeError("Insufficient GPU memory for STTN model")
            
            # Create model
            model = STTNModel(self.model_config)
            model = model.to(self.gpu_env.device)
            
            # Load pretrained weights if available
            if model_path and Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location=self.gpu_env.device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print(f"Loaded STTN model from {model_path}")
            else:
                print("Using randomly initialized STTN model (no pretrained weights)")
            
            # Optimize for inference
            model.eval()
            
            # Enable optimizations
            if hasattr(torch, 'compile') and self.gpu_env.device.type == 'cuda':
                try:
                    model = torch.compile(model, mode='reduce-overhead')
                    print("Enabled torch.compile optimization for STTN")
                except Exception as e:
                    print(f"Failed to compile STTN model: {e}")
            
            # Enable mixed precision if supported
            if self.gpu_env.device.type == 'cuda':
                torch.backends.cudnn.benchmark = True
                torch.set_float32_matmul_precision('high')
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load STTN model: {e}")
    
    def preprocess_frames(self, frames: np.ndarray) -> torch.Tensor:
        """
        Preprocess video frames for STTN model.
        
        Args:
            frames: Input frames [T, H, W, C] in uint8 format
            
        Returns:
            Preprocessed frames [1, T, C, H, W] in float32 format
        """
        # Convert to float and normalize to [0, 1]
        frames = frames.astype(np.float32) / 255.0
        
        # Convert to tensor and rearrange dimensions
        frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, C, H, W]
        frames_tensor = frames_tensor.unsqueeze(0)  # [1, T, C, H, W]
        
        return frames_tensor.to(self.gpu_env.device)
    
    def create_subtitle_mask(self, frames_shape: Tuple[int, ...], subtitle_area: Dict[str, int]) -> torch.Tensor:
        """
        Create binary mask for subtitle area.
        
        Args:
            frames_shape: Shape of input frames [T, H, W, C]
            subtitle_area: Dictionary with x1, y1, x2, y2 coordinates
            
        Returns:
            Binary mask [1, T, 1, H, W]
        """
        T, H, W, C = frames_shape
        
        # Create mask
        mask = np.zeros((T, H, W), dtype=np.float32)
        
        # Fill subtitle area
        y1, y2 = subtitle_area['y1'], subtitle_area['y2']
        x1, x2 = subtitle_area['x1'], subtitle_area['x2']
        
        # Ensure coordinates are within bounds
        y1, y2 = max(0, y1), min(H, y2)
        x1, x2 = max(0, x1), min(W, x2)
        
        mask[:, y1:y2, x1:x2] = 1.0
        
        # Convert to tensor
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(2)  # [1, T, 1, H, W]
        
        return mask_tensor.to(self.gpu_env.device)
    
    def postprocess_frames(self, frames: torch.Tensor) -> np.ndarray:
        """
        Postprocess output frames from STTN model.
        
        Args:
            frames: Output frames [1, T, C, H, W] in float32 format
            
        Returns:
            Postprocessed frames [T, H, W, C] in uint8 format
        """
        # Remove batch dimension and rearrange
        frames = frames.squeeze(0).permute(1, 2, 3, 0)  # [T, H, W, C]
        
        # Clamp to [0, 1] and convert to uint8
        frames = torch.clamp(frames, 0, 1)
        frames = (frames * 255).cpu().numpy().astype(np.uint8)
        
        return frames
    
    async def inpaint_video(self, frames: np.ndarray, subtitle_area: Dict[str, int]) -> np.ndarray:
        """
        Perform video inpainting using STTN model.
        
        Args:
            frames: Input video frames [T, H, W, C]
            subtitle_area: Subtitle area coordinates
            
        Returns:
            Inpainted video frames [T, H, W, C]
        """
        if not hasattr(self, '_model') or self._model is None:
            raise RuntimeError("STTN model not loaded")
        
        try:
            with torch.no_grad():
                # Preprocess frames
                frames_tensor = self.preprocess_frames(frames)
                
                # Create subtitle mask
                mask_tensor = self.create_subtitle_mask(frames.shape, subtitle_area)
                
                # Process in chunks to manage memory
                chunk_size = self.model_config["num_frames"]
                T = frames_tensor.size(1)
                output_frames = []
                
                for i in range(0, T, chunk_size):
                    end_idx = min(i + chunk_size, T)
                    
                    # Get chunk
                    frame_chunk = frames_tensor[:, i:end_idx]
                    mask_chunk = mask_tensor[:, i:end_idx]
                    
                    # Pad if necessary
                    if frame_chunk.size(1) < chunk_size:
                        pad_size = chunk_size - frame_chunk.size(1)
                        frame_chunk = torch.cat([
                            frame_chunk,
                            frame_chunk[:, -1:].repeat(1, pad_size, 1, 1, 1)
                        ], dim=1)
                        mask_chunk = torch.cat([
                            mask_chunk,
                            mask_chunk[:, -1:].repeat(1, pad_size, 1, 1, 1)
                        ], dim=1)
                    
                    # Inpaint chunk
                    with torch.cuda.amp.autocast(enabled=self.gpu_env.device.type == 'cuda'):
                        inpainted_chunk = self._model(frame_chunk, mask_chunk)
                    
                    # Take only the needed frames
                    needed_frames = end_idx - i
                    output_frames.append(inpainted_chunk[:, :needed_frames])
                
                # Concatenate all chunks
                output_tensor = torch.cat(output_frames, dim=1)
                
                # Postprocess
                result = self.postprocess_frames(output_tensor)
                
                return result
                
        except Exception as e:
            raise RuntimeError(f"STTN inpainting failed: {e}")
