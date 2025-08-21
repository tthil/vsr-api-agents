"""
Video I/O operations for reading and writing video files.
Handles frame extraction, video metadata, and efficient streaming.
"""
import asyncio
import logging
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Tuple
from dataclasses import dataclass
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import ffmpeg


logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Video metadata container."""
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    codec: str
    bitrate: Optional[int] = None
    format: Optional[str] = None


class VideoReader:
    """
    Async video reader for efficient frame extraction.
    """
    
    def __init__(self, video_path: Path, max_workers: int = 4):
        self.video_path = video_path
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._metadata: Optional[VideoMetadata] = None
        self._cap: Optional[cv2.VideoCapture] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def initialize(self):
        """Initialize video reader."""
        try:
            # Initialize video capture in thread pool
            loop = asyncio.get_event_loop()
            self._cap = await loop.run_in_executor(
                self.executor,
                cv2.VideoCapture,
                str(self.video_path)
            )
            
            if not self._cap.isOpened():
                raise ValueError(f"Could not open video file: {self.video_path}")
            
            logger.info(f"Initialized video reader for: {self.video_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize video reader: {e}")
            raise
    
    async def get_metadata(self) -> VideoMetadata:
        """
        Extract video metadata.
        
        Returns:
            VideoMetadata object with video information
        """
        if self._metadata:
            return self._metadata
        
        if not self._cap:
            await self.initialize()
        
        try:
            loop = asyncio.get_event_loop()
            
            # Get metadata in thread pool
            metadata_dict = await loop.run_in_executor(
                self.executor,
                self._extract_metadata
            )
            
            self._metadata = VideoMetadata(**metadata_dict)
            return self._metadata
            
        except Exception as e:
            logger.error(f"Failed to extract metadata: {e}")
            raise
    
    def _extract_metadata(self) -> dict:
        """Extract metadata using OpenCV (runs in thread pool)."""
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Get codec information
        fourcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
            "codec": codec
        }
    
    async def read_frame(self) -> Optional[np.ndarray]:
        """
        Read a single frame from the video.
        
        Returns:
            Frame as numpy array or None if end of video
        """
        if not self._cap:
            await self.initialize()
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._read_frame_sync
        )
    
    def _read_frame_sync(self) -> Optional[np.ndarray]:
        """Read frame synchronously (runs in thread pool)."""
        ret, frame = self._cap.read()
        return frame if ret else None
    
    async def read_frames_batch(self, batch_size: int = 8) -> AsyncGenerator[List[np.ndarray], None]:
        """
        Read frames in batches for efficient processing.
        
        Args:
            batch_size: Number of frames per batch
            
        Yields:
            Batches of frames as numpy arrays
        """
        if not self._cap:
            await self.initialize()
        
        batch = []
        
        while True:
            frame = await self.read_frame()
            if frame is None:
                # End of video - yield remaining frames if any
                if batch:
                    yield batch
                break
            
            batch.append(frame)
            
            if len(batch) >= batch_size:
                yield batch
                batch = []
    
    async def seek_frame(self, frame_number: int) -> bool:
        """
        Seek to a specific frame.
        
        Args:
            frame_number: Frame number to seek to
            
        Returns:
            True if seek was successful
        """
        if not self._cap:
            await self.initialize()
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        )
    
    async def close(self):
        """Close video reader and cleanup resources."""
        if self._cap:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self._cap.release
            )
            self._cap = None
        
        self.executor.shutdown(wait=True)
        logger.info(f"Closed video reader for: {self.video_path}")


class VideoWriter:
    """
    Async video writer for efficient video creation.
    """
    
    def __init__(self, 
                 output_path: Path,
                 fps: float,
                 frame_size: Tuple[int, int],
                 codec: str = 'mp4v',
                 max_workers: int = 2):
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.codec = codec
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._writer: Optional[cv2.VideoWriter] = None
        self._frame_buffer: List[np.ndarray] = []
        self._buffer_size = 30  # Buffer 30 frames before writing
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.finalize()
    
    async def initialize(self):
        """Initialize video writer."""
        try:
            # Create output directory if it doesn't exist
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Initialize video writer in thread pool
            loop = asyncio.get_event_loop()
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            
            self._writer = await loop.run_in_executor(
                self.executor,
                lambda: cv2.VideoWriter(
                    str(self.output_path),
                    fourcc,
                    self.fps,
                    self.frame_size
                )
            )
            
            if not self._writer.isOpened():
                raise ValueError(f"Could not initialize video writer: {self.output_path}")
            
            logger.info(f"Initialized video writer for: {self.output_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize video writer: {e}")
            raise
    
    async def write_frame(self, frame: np.ndarray):
        """
        Write a single frame to the video.
        
        Args:
            frame: Frame to write as numpy array
        """
        if not self._writer:
            await self.initialize()
        
        # Add frame to buffer
        self._frame_buffer.append(frame.copy())
        
        # Write buffer when it's full
        if len(self._frame_buffer) >= self._buffer_size:
            await self._flush_buffer()
    
    async def write_frames_batch(self, frames: List[np.ndarray]):
        """
        Write multiple frames efficiently.
        
        Args:
            frames: List of frames to write
        """
        for frame in frames:
            await self.write_frame(frame)
    
    async def _flush_buffer(self):
        """Flush frame buffer to video file."""
        if not self._frame_buffer:
            return
        
        frames_to_write = self._frame_buffer.copy()
        self._frame_buffer.clear()
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            self._write_frames_sync,
            frames_to_write
        )
    
    def _write_frames_sync(self, frames: List[np.ndarray]):
        """Write frames synchronously (runs in thread pool)."""
        for frame in frames:
            self._writer.write(frame)
    
    async def finalize(self):
        """Finalize video writing and cleanup resources."""
        # Flush remaining frames
        await self._flush_buffer()
        
        if self._writer:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self._writer.release
            )
            self._writer = None
        
        self.executor.shutdown(wait=True)
        logger.info(f"Finalized video writer for: {self.output_path}")


class FFmpegProcessor:
    """
    FFmpeg-based video processing for advanced operations.
    """
    
    @staticmethod
    async def convert_video(
        input_path: Path,
        output_path: Path,
        codec: str = 'libx264',
        crf: int = 23,
        preset: str = 'medium'
    ) -> bool:
        """
        Convert video using FFmpeg.
        
        Args:
            input_path: Input video path
            output_path: Output video path
            codec: Video codec to use
            crf: Constant Rate Factor (quality)
            preset: Encoding preset
            
        Returns:
            True if conversion was successful
        """
        try:
            # Create output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Run FFmpeg conversion
            stream = ffmpeg.input(str(input_path))
            stream = ffmpeg.output(
                stream,
                str(output_path),
                vcodec=codec,
                crf=crf,
                preset=preset,
                movflags='faststart'  # Optimize for web streaming
            )
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: ffmpeg.run(stream, overwrite_output=True, quiet=True)
            )
            
            logger.info(f"Successfully converted video: {input_path} -> {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"FFmpeg conversion failed: {e}")
            return False
    
    @staticmethod
    async def extract_metadata_ffmpeg(video_path: Path) -> dict:
        """
        Extract detailed metadata using FFmpeg.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with detailed metadata
        """
        try:
            loop = asyncio.get_event_loop()
            probe = await loop.run_in_executor(
                None,
                lambda: ffmpeg.probe(str(video_path))
            )
            
            video_stream = next(
                (stream for stream in probe['streams'] if stream['codec_type'] == 'video'),
                None
            )
            
            if not video_stream:
                raise ValueError("No video stream found")
            
            return {
                'width': int(video_stream['width']),
                'height': int(video_stream['height']),
                'fps': eval(video_stream['r_frame_rate']),
                'duration': float(video_stream.get('duration', 0)),
                'codec': video_stream['codec_name'],
                'bitrate': int(video_stream.get('bit_rate', 0)),
                'format': probe['format']['format_name']
            }
            
        except Exception as e:
            logger.error(f"FFmpeg metadata extraction failed: {e}")
            raise
    
    @staticmethod
    async def optimize_for_web(input_path: Path, output_path: Path) -> bool:
        """
        Optimize video for web delivery.
        
        Args:
            input_path: Input video path
            output_path: Output video path
            
        Returns:
            True if optimization was successful
        """
        try:
            # Create output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            stream = ffmpeg.input(str(input_path))
            stream = ffmpeg.output(
                stream,
                str(output_path),
                vcodec='libx264',
                acodec='aac',
                crf=28,  # Slightly higher compression for web
                preset='fast',
                movflags='faststart',
                pix_fmt='yuv420p',  # Ensure compatibility
                vf='scale=trunc(iw/2)*2:trunc(ih/2)*2'  # Ensure even dimensions
            )
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: ffmpeg.run(stream, overwrite_output=True, quiet=True)
            )
            
            logger.info(f"Successfully optimized video for web: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Web optimization failed: {e}")
            return False


# Utility functions
async def validate_video_file(video_path: Path) -> bool:
    """
    Validate that a file is a valid video.
    
    Args:
        video_path: Path to video file
        
    Returns:
        True if file is a valid video
    """
    try:
        async with VideoReader(video_path) as reader:
            metadata = await reader.get_metadata()
            return metadata.frame_count > 0 and metadata.fps > 0
    except Exception:
        return False


async def get_video_thumbnail(video_path: Path, timestamp: float = 1.0) -> Optional[np.ndarray]:
    """
    Extract a thumbnail from video at specified timestamp.
    
    Args:
        video_path: Path to video file
        timestamp: Timestamp in seconds
        
    Returns:
        Thumbnail frame or None if failed
    """
    try:
        async with VideoReader(video_path) as reader:
            metadata = await reader.get_metadata()
            frame_number = int(timestamp * metadata.fps)
            
            if await reader.seek_frame(frame_number):
                return await reader.read_frame()
    except Exception as e:
        logger.error(f"Failed to extract thumbnail: {e}")
    
    return None
