"""
ProPainter model implementation for video inpainting.
Advanced temporal consistency with flow-guided propagation.
Optimized for H100/A100 GPUs with efficient memory management.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import cv2
from pathlib import Path

from vsr_worker.gpu.environment import GPUEnvironment
from vsr_worker.gpu.model_loader import BaseModelLoader


class FlowEstimator(nn.Module):
    """Optical flow estimation network."""
    
    def __init__(self, in_channels: int = 6):
        super().__init__()
        
        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 7, padding=3),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Flow prediction
        self.flow_pred = nn.Conv2d(64, 2, 3, padding=1)
        
    def forward(self, frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
        """
        Estimate optical flow between two frames.
        
        Args:
            frame1: First frame [B, 3, H, W]
            frame2: Second frame [B, 3, H, W]
            
        Returns:
            Optical flow [B, 2, H, W]
        """
        # Concatenate frames
        input_tensor = torch.cat([frame1, frame2], dim=1)
        
        # Encoder
        conv1 = self.conv1(input_tensor)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        
        # Decoder with skip connections
        upconv3 = self.upconv3(conv4)
        upconv3 = torch.cat([upconv3, conv3], dim=1)
        
        upconv2 = self.upconv2(upconv3)
        upconv2 = torch.cat([upconv2, conv2], dim=1)
        
        upconv1 = self.upconv1(upconv2)
        upconv1 = torch.cat([upconv1, conv1], dim=1)
        
        # Flow prediction
        flow = self.flow_pred(upconv1)
        
        return flow


class TemporalPropagationModule(nn.Module):
    """Temporal propagation module for consistent inpainting."""
    
    def __init__(self, feature_channels: int = 256):
        super().__init__()
        
        self.feature_channels = feature_channels
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, feature_channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Propagation network
        self.propagation_net = nn.Sequential(
            nn.Conv2d(feature_channels * 2, feature_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Feature decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_channels, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )
    
    def warp_features(self, features: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Warp features using optical flow."""
        B, C, H, W = features.shape
        
        # Create coordinate grid
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=features.device, dtype=torch.float32),
            torch.arange(W, device=features.device, dtype=torch.float32),
            indexing='ij'
        )
        
        # Normalize coordinates to [-1, 1]
        x_coords = (x_coords / (W - 1)) * 2 - 1
        y_coords = (y_coords / (H - 1)) * 2 - 1
        
        # Add flow
        flow_x = flow[:, 0] / (W - 1) * 2  # Normalize flow
        flow_y = flow[:, 1] / (H - 1) * 2
        
        x_coords = x_coords.unsqueeze(0).expand(B, -1, -1) + flow_x
        y_coords = y_coords.unsqueeze(0).expand(B, -1, -1) + flow_y
        
        # Stack coordinates
        grid = torch.stack([x_coords, y_coords], dim=-1)  # [B, H, W, 2]
        
        # Warp features
        warped_features = F.grid_sample(
            features, grid, mode='bilinear', padding_mode='border', align_corners=True
        )
        
        return warped_features
    
    def forward(self, current_frame: torch.Tensor, reference_frame: torch.Tensor, 
                flow: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Propagate information from reference frame to current frame.
        
        Args:
            current_frame: Current frame to inpaint [B, 3, H, W]
            reference_frame: Reference frame [B, 3, H, W]
            flow: Optical flow from reference to current [B, 2, H, W]
            mask: Inpainting mask [B, 1, H, W]
            
        Returns:
            Propagated frame [B, 3, H, W]
        """
        # Extract features
        current_features = self.feature_extractor(current_frame)
        reference_features = self.feature_extractor(reference_frame)
        
        # Warp reference features
        warped_features = self.warp_features(reference_features, flow)
        
        # Combine features
        combined_features = torch.cat([current_features, warped_features], dim=1)
        propagated_features = self.propagation_net(combined_features)
        
        # Decode to image
        propagated_frame = self.decoder(propagated_features)
        propagated_frame = (propagated_frame + 1) / 2  # Convert to [0, 1]
        
        # Downsample mask to match flow resolution
        mask_downsampled = F.interpolate(mask, size=flow.shape[2:], mode='nearest')
        mask_upsampled = F.interpolate(mask_downsampled, size=current_frame.shape[2:], mode='nearest')
        
        # Blend with original frame
        result = current_frame * (1 - mask_upsampled) + propagated_frame * mask_upsampled
        
        return result


class ProPainterModel(nn.Module):
    """ProPainter model for video inpainting with temporal consistency."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Model components
        self.flow_estimator = FlowEstimator()
        self.temporal_propagation = TemporalPropagationModule(
            feature_channels=config.get("feature_channels", 256)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, frames: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ProPainter model.
        
        Args:
            frames: Input video frames [B, T, 3, H, W]
            masks: Binary masks [B, T, 1, H, W]
            
        Returns:
            Inpainted frames [B, T, 3, H, W]
        """
        B, T, C, H, W = frames.shape
        
        # Process frames sequentially with temporal propagation
        output_frames = []
        
        for t in range(T):
            current_frame = frames[:, t]
            current_mask = masks[:, t]
            
            if t == 0:
                # For first frame, use simple inpainting (could be improved)
                inpainted_frame = self._simple_inpaint(current_frame, current_mask)
            else:
                # Use temporal propagation from previous frame
                prev_frame = output_frames[-1]
                
                # Estimate optical flow
                flow = self.flow_estimator(prev_frame, current_frame)
                
                # Propagate information
                inpainted_frame = self.temporal_propagation(
                    current_frame, prev_frame, flow, current_mask
                )
            
            output_frames.append(inpainted_frame)
        
        # Stack output frames
        output = torch.stack(output_frames, dim=1)
        
        return output
    
    def _simple_inpaint(self, frame: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Simple inpainting for the first frame using nearest neighbor."""
        # This is a placeholder - in practice, you'd use a more sophisticated method
        # For now, just fill with nearest valid pixels
        
        # Convert to numpy for OpenCV processing
        frame_np = frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
        mask_np = mask.squeeze(0).squeeze(0).cpu().numpy()
        
        # Convert to uint8
        frame_np = (frame_np * 255).astype(np.uint8)
        mask_np = (mask_np * 255).astype(np.uint8)
        
        # Use OpenCV inpainting as fallback
        inpainted_np = cv2.inpaint(frame_np, mask_np, 3, cv2.INPAINT_TELEA)
        
        # Convert back to tensor
        inpainted = torch.from_numpy(inpainted_np.astype(np.float32) / 255.0)
        inpainted = inpainted.permute(2, 0, 1).unsqueeze(0).to(frame.device)
        
        return inpainted


class ProPainterModelLoader(BaseModelLoader):
    """Model loader for ProPainter model."""
    
    def __init__(self, gpu_env: GPUEnvironment):
        super().__init__(gpu_env)
        self.model_name = "ProPainter"
        self.model_config = {
            "feature_channels": 256
        }
    
    async def load_model(self, model_path: Optional[str] = None) -> ProPainterModel:
        """Load ProPainter model with optimizations."""
        try:
            # Check GPU memory (ProPainter requires more memory)
            if not self.gpu_env.check_memory_available(3072):  # 3GB minimum
                raise RuntimeError("Insufficient GPU memory for ProPainter model")
            
            # Create model
            model = ProPainterModel(self.model_config)
            model = model.to(self.gpu_env.device)
            
            # Load pretrained weights if available
            if model_path and Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location=self.gpu_env.device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print(f"Loaded ProPainter model from {model_path}")
            else:
                print("Using randomly initialized ProPainter model (no pretrained weights)")
            
            # Optimize for inference
            model.eval()
            
            # Enable optimizations
            if hasattr(torch, 'compile') and self.gpu_env.device.type == 'cuda':
                try:
                    model = torch.compile(model, mode='reduce-overhead')
                    print("Enabled torch.compile optimization for ProPainter")
                except Exception as e:
                    print(f"Failed to compile ProPainter model: {e}")
            
            # Enable mixed precision if supported
            if self.gpu_env.device.type == 'cuda':
                torch.backends.cudnn.benchmark = True
                torch.set_float32_matmul_precision('high')
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load ProPainter model: {e}")
    
    def preprocess_frames(self, frames: np.ndarray) -> torch.Tensor:
        """
        Preprocess video frames for ProPainter model.
        
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
        Postprocess output frames from ProPainter model.
        
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
        Perform video inpainting using ProPainter model.
        
        Args:
            frames: Input video frames [T, H, W, C]
            subtitle_area: Subtitle area coordinates
            
        Returns:
            Inpainted video frames [T, H, W, C]
        """
        if not hasattr(self, '_model') or self._model is None:
            raise RuntimeError("ProPainter model not loaded")
        
        try:
            with torch.no_grad():
                # Preprocess frames
                frames_tensor = self.preprocess_frames(frames)
                
                # Create subtitle mask
                mask_tensor = self.create_subtitle_mask(frames.shape, subtitle_area)
                
                # Process video in chunks to manage memory
                chunk_size = 8  # Process 8 frames at a time
                T = frames_tensor.size(1)
                output_chunks = []
                
                for i in range(0, T, chunk_size):
                    end_idx = min(i + chunk_size, T)
                    
                    # Get chunk
                    frame_chunk = frames_tensor[:, i:end_idx]
                    mask_chunk = mask_tensor[:, i:end_idx]
                    
                    # Inpaint chunk
                    with torch.cuda.amp.autocast(enabled=self.gpu_env.device.type == 'cuda'):
                        inpainted_chunk = self._model(frame_chunk, mask_chunk)
                    
                    output_chunks.append(inpainted_chunk)
                
                # Concatenate all chunks
                output_tensor = torch.cat(output_chunks, dim=1)
                
                # Postprocess
                result = self.postprocess_frames(output_tensor)
                
                return result
                
        except Exception as e:
            raise RuntimeError(f"ProPainter inpainting failed: {e}")
    
    def estimate_processing_time(self, num_frames: int, resolution: Tuple[int, int]) -> float:
        """
        Estimate processing time for ProPainter model.
        
        Args:
            num_frames: Number of frames to process
            resolution: Frame resolution (width, height)
            
        Returns:
            Estimated processing time in seconds
        """
        # Base time per frame (ProPainter is slower due to temporal consistency)
        base_time_per_frame = 0.3  # seconds
        
        # Resolution factor (normalized to 1080p)
        width, height = resolution
        resolution_factor = (width * height) / (1920 * 1080)
        
        # GPU factor (H100 is faster than A100)
        gpu_factor = 0.6 if "H100" in str(self.gpu_env.device) else 1.0
        
        total_time = num_frames * base_time_per_frame * resolution_factor * gpu_factor
        
        return max(total_time, 2.0)  # Minimum 2 seconds
