"""
CPU-optimized model implementations for local development.

Provides lightweight versions of STTN, LAMA, and ProPainter models
optimized for CPU inference with reduced complexity and faster processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from vsr_shared.logging import get_logger

logger = get_logger(__name__)


class CPUOptimizedSTTN(nn.Module):
    """CPU-optimized STTN model with reduced complexity."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.num_frames = config.get("num_frames", 3)
        self.img_size = config.get("img_size", (216, 120))
        self.patch_size = config.get("patch_size", 12)
        self.embed_dim = config.get("embed_dim", 128)
        self.num_heads = config.get("num_heads", 4)
        self.num_layers = config.get("num_layers", 2)
        
        # Simplified architecture for CPU
        self.patch_embed = PatchEmbedding(
            img_size=self.img_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim
        )
        
        self.transformer = SimplifiedTransformer(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers
        )
        
        self.output_proj = nn.Linear(self.embed_dim, 3 * self.patch_size * self.patch_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, frames: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for CPU-optimized STTN.
        
        Args:
            frames: Input frames [B, T, C, H, W]
            mask: Mask tensor [B, T, 1, H, W]
            
        Returns:
            Inpainted frames [B, T, C, H, W]
        """
        B, T, C, H, W = frames.shape
        
        # Process frames sequentially to save memory
        output_frames = []
        
        for t in range(T):
            frame = frames[:, t]  # [B, C, H, W]
            frame_mask = mask[:, t]  # [B, 1, H, W]
            
            # Patch embedding
            patches = self.patch_embed(frame)  # [B, N, D]
            
            # Add positional encoding (simplified)
            patches = patches + self._get_pos_encoding(patches.shape[1], self.embed_dim)
            
            # Transformer processing
            features = self.transformer(patches)
            
            # Output projection
            output = self.output_proj(features)
            
            # Reconstruct frame
            reconstructed = self._reconstruct_frame(output, frame.shape)
            
            # Apply mask (keep original pixels where mask is 0)
            mask_expanded = frame_mask.expand_as(reconstructed)
            result = frame * (1 - mask_expanded) + reconstructed * mask_expanded
            
            output_frames.append(result)
        
        return torch.stack(output_frames, dim=1)
    
    def _get_pos_encoding(self, seq_len: int, embed_dim: int) -> torch.Tensor:
        """Simple positional encoding."""
        pos = torch.arange(seq_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           -(np.log(10000.0) / embed_dim))
        
        pe = torch.zeros(seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        
        return pe.unsqueeze(0)
    
    def _reconstruct_frame(self, patches: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
        """Reconstruct frame from patches."""
        B, C, H, W = target_shape
        
        # Reshape patches
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size
        
        patches = patches.view(B, patch_h, patch_w, 3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 3, 1, 4, 2, 5)
        patches = patches.contiguous().view(B, 3, H, W)
        
        return patches


class CPUOptimizedLAMA(nn.Module):
    """CPU-optimized LAMA model with simplified architecture."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.in_channels = config.get("in_channels", 4)  # RGB + mask
        self.base_channels = config.get("base_channels", 32)
        self.num_blocks = config.get("num_blocks", 4)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, self.base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.base_channels, self.base_channels * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(self.base_channels * 2)
            for _ in range(self.num_blocks)
        ])
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.base_channels * 2, self.base_channels, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.base_channels, 3, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for CPU-optimized LAMA.
        
        Args:
            x: Input tensor [B, 4, H, W] (RGB + mask)
            
        Returns:
            Inpainted image [B, 3, H, W]
        """
        # Encoder
        features = self.encoder(x)
        
        # Residual blocks
        for block in self.res_blocks:
            features = block(features)
        
        # Decoder
        output = self.decoder(features)
        
        return output


class CPUOptimizedProPainter(nn.Module):
    """CPU-optimized ProPainter using OpenCV fallback."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.fallback_only = config.get("fallback_only", True)
        
        if not self.fallback_only:
            # Simple CNN for basic processing
            self.conv_layers = nn.Sequential(
                nn.Conv2d(4, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 3, 3, padding=1),
                nn.Sigmoid()
            )
    
    def forward(self, frames: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for CPU-optimized ProPainter.
        
        Args:
            frames: Input frames [B, T, C, H, W]
            mask: Mask tensor [B, T, 1, H, W]
            
        Returns:
            Inpainted frames [B, T, C, H, W]
        """
        if self.fallback_only:
            return self._opencv_fallback(frames, mask)
        else:
            return self._simple_cnn_processing(frames, mask)
    
    def _opencv_fallback(self, frames: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Use OpenCV inpainting as fallback."""
        B, T, C, H, W = frames.shape
        output_frames = []
        
        for b in range(B):
            batch_frames = []
            for t in range(T):
                # Convert to numpy
                frame_np = frames[b, t].permute(1, 2, 0).cpu().numpy()
                mask_np = mask[b, t, 0].cpu().numpy()
                
                # Convert to uint8
                frame_np = (frame_np * 255).astype(np.uint8)
                mask_np = (mask_np * 255).astype(np.uint8)
                
                # OpenCV inpainting
                inpainted = cv2.inpaint(frame_np, mask_np, 3, cv2.INPAINT_TELEA)
                
                # Convert back to tensor
                inpainted_tensor = torch.from_numpy(inpainted.astype(np.float32) / 255.0)
                inpainted_tensor = inpainted_tensor.permute(2, 0, 1)
                
                batch_frames.append(inpainted_tensor)
            
            output_frames.append(torch.stack(batch_frames))
        
        return torch.stack(output_frames)
    
    def _simple_cnn_processing(self, frames: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Simple CNN processing for each frame."""
        B, T, C, H, W = frames.shape
        output_frames = []
        
        for t in range(T):
            frame = frames[:, t]  # [B, C, H, W]
            frame_mask = mask[:, t]  # [B, 1, H, W]
            
            # Concatenate frame and mask
            input_tensor = torch.cat([frame, frame_mask], dim=1)
            
            # Process with CNN
            output = self.conv_layers(input_tensor)
            
            # Apply mask
            mask_expanded = frame_mask.expand_as(output)
            result = frame * (1 - mask_expanded) + output * mask_expanded
            
            output_frames.append(result)
        
        return torch.stack(output_frames, dim=1)


class PatchEmbedding(nn.Module):
    """Patch embedding for CPU-optimized STTN."""
    
    def __init__(self, img_size: Tuple[int, int], patch_size: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to patch embeddings."""
        B, C, H, W = x.shape
        
        # Project to patches
        patches = self.proj(x)  # [B, embed_dim, H//patch_size, W//patch_size]
        
        # Flatten spatial dimensions
        patches = patches.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        return patches


class SimplifiedTransformer(nn.Module):
    """Simplified transformer for CPU processing."""
    
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            SimplifiedTransformerLayer(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer layers."""
        for layer in self.layers:
            x = layer(x)
        return x


class SimplifiedTransformerLayer(nn.Module):
    """Simplified transformer layer."""
    
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 2, embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer layer."""
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # MLP
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        
        return x


class ResidualBlock(nn.Module):
    """Residual block for CPU-optimized LAMA."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block."""
        residual = x
        
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        
        out += residual
        out = self.relu(out)
        
        return out


class CPUModelQuantizer:
    """Utility for quantizing models for CPU inference."""
    
    @staticmethod
    def quantize_model(model: nn.Module, quantization_type: str = "dynamic") -> nn.Module:
        """
        Quantize model for CPU inference.
        
        Args:
            model: PyTorch model to quantize
            quantization_type: Type of quantization (dynamic, static)
            
        Returns:
            Quantized model
        """
        try:
            if quantization_type == "dynamic":
                # Dynamic quantization
                quantized_model = torch.quantization.quantize_dynamic(
                    model, 
                    {nn.Linear, nn.Conv2d}, 
                    dtype=torch.qint8
                )
            else:
                # Static quantization (requires calibration)
                model.eval()
                model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                torch.quantization.prepare(model, inplace=True)
                # Note: Would need calibration data here for static quantization
                quantized_model = torch.quantization.convert(model, inplace=False)
            
            logger.info(f"Model quantized using {quantization_type} quantization")
            return quantized_model
            
        except Exception as e:
            logger.warning(f"Quantization failed: {e}, returning original model")
            return model
    
    @staticmethod
    def get_model_size(model: nn.Module) -> Dict[str, float]:
        """Get model size information."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size = param_size + buffer_size
        
        return {
            "parameters_mb": param_size / 1024**2,
            "buffers_mb": buffer_size / 1024**2,
            "total_mb": total_size / 1024**2
        }


def create_cpu_optimized_model(model_type: str, config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create CPU-optimized models.
    
    Args:
        model_type: Type of model (sttn, lama, propainter)
        config: Model configuration
        
    Returns:
        CPU-optimized model instance
    """
    model_type = model_type.lower()
    
    if model_type == "sttn":
        model = CPUOptimizedSTTN(config)
    elif model_type == "lama":
        model = CPUOptimizedLAMA(config)
    elif model_type == "propainter":
        model = CPUOptimizedProPainter(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Apply quantization if requested
    if config.get("quantization", False):
        model = CPUModelQuantizer.quantize_model(model)
    
    # Set to evaluation mode
    model.eval()
    
    logger.info(f"Created CPU-optimized {model_type} model")
    return model
