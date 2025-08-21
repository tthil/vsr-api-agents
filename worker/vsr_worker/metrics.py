"""
Worker metrics and GPU monitoring for VSR worker.

Tracks job processing performance, GPU utilization, and failure categorization.
"""

import asyncio
import time
import subprocess
import json
import psutil
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class JobMetrics:
    """Metrics for a single job processing."""
    job_id: str
    start_time: float
    end_time: Optional[float] = None
    processing_time: Optional[float] = None
    model_time_per_frame: Optional[float] = None
    frames_processed: int = 0
    gpu_memory_peak: Optional[float] = None
    success: bool = False
    failure_reason: Optional[str] = None


@dataclass
class GPUMetrics:
    """GPU utilization metrics."""
    gpu_id: int
    utilization_percent: float
    memory_used_mb: float
    memory_total_mb: float
    memory_percent: float
    temperature_c: float
    power_draw_w: float


class WorkerMetricsCollector:
    """
    Collects and tracks worker performance metrics.
    
    Monitors job processing times, GPU utilization, memory usage,
    and categorizes failures for operational insights.
    """
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        
        # Job metrics storage
        self.job_metrics: deque = deque(maxlen=max_history)
        self.active_jobs: Dict[str, JobMetrics] = {}
        
        # Performance counters
        self.jobs_completed = 0
        self.jobs_failed = 0
        self.total_processing_time = 0.0
        self.total_frames_processed = 0
        
        # Failure categorization
        self.failure_reasons = defaultdict(int)
        
        # GPU monitoring
        self.gpu_available = self._check_gpu_availability()
        self.gpu_metrics_history: deque = deque(maxlen=100)  # Last 100 readings
        
        # System metrics
        self.start_time = time.time()
        
    def start_job(self, job_id: str) -> JobMetrics:
        """Start tracking metrics for a job."""
        job_metrics = JobMetrics(
            job_id=job_id,
            start_time=time.time()
        )
        
        self.active_jobs[job_id] = job_metrics
        
        logger.info(
            "Job metrics tracking started",
            job_id=job_id,
            active_jobs=len(self.active_jobs)
        )
        
        return job_metrics
    
    def complete_job(self, job_id: str, frames_processed: int = 0, 
                    model_time_per_frame: float = None, success: bool = True,
                    failure_reason: str = None):
        """Complete job metrics tracking."""
        if job_id not in self.active_jobs:
            logger.warning("Job metrics not found", job_id=job_id)
            return
        
        job_metrics = self.active_jobs[job_id]
        job_metrics.end_time = time.time()
        job_metrics.processing_time = job_metrics.end_time - job_metrics.start_time
        job_metrics.frames_processed = frames_processed
        job_metrics.model_time_per_frame = model_time_per_frame
        job_metrics.success = success
        job_metrics.failure_reason = failure_reason
        
        # Record GPU memory peak if available
        if self.gpu_available:
            try:
                gpu_info = self._get_gpu_info()
                if gpu_info:
                    job_metrics.gpu_memory_peak = gpu_info[0].memory_used_mb
            except Exception:
                pass
        
        # Update counters
        if success:
            self.jobs_completed += 1
            self.total_processing_time += job_metrics.processing_time
            self.total_frames_processed += frames_processed
        else:
            self.jobs_failed += 1
            if failure_reason:
                self.failure_reasons[failure_reason] += 1
        
        # Store in history
        self.job_metrics.append(job_metrics)
        
        # Remove from active jobs
        del self.active_jobs[job_id]
        
        logger.info(
            "Job metrics completed",
            job_id=job_id,
            processing_time=job_metrics.processing_time,
            frames_processed=frames_processed,
            success=success,
            failure_reason=failure_reason
        )
    
    def get_gpu_metrics(self) -> List[GPUMetrics]:
        """Get current GPU metrics."""
        if not self.gpu_available:
            return []
        
        try:
            return self._get_gpu_info()
        except Exception as e:
            logger.error("Failed to get GPU metrics", error=str(e))
            return []
    
    def record_gpu_metrics(self):
        """Record current GPU metrics to history."""
        gpu_metrics = self.get_gpu_metrics()
        if gpu_metrics:
            self.gpu_metrics_history.append({
                'timestamp': time.time(),
                'metrics': gpu_metrics
            })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Calculate averages
        avg_processing_time = 0
        avg_frames_per_second = 0
        avg_model_time_per_frame = 0
        
        if self.jobs_completed > 0:
            avg_processing_time = self.total_processing_time / self.jobs_completed
            
            if self.total_frames_processed > 0:
                avg_frames_per_second = self.total_frames_processed / self.total_processing_time
                
                # Calculate average model time per frame from recent jobs
                recent_jobs = [job for job in self.job_metrics 
                             if job.model_time_per_frame is not None][-10:]
                if recent_jobs:
                    avg_model_time_per_frame = sum(job.model_time_per_frame for job in recent_jobs) / len(recent_jobs)
        
        # Recent performance (last hour)
        one_hour_ago = current_time - 3600
        recent_jobs = [job for job in self.job_metrics 
                      if job.end_time and job.end_time > one_hour_ago]
        
        recent_completed = len([job for job in recent_jobs if job.success])
        recent_failed = len([job for job in recent_jobs if not job.success])
        
        # GPU metrics summary
        gpu_summary = {}
        if self.gpu_available and self.gpu_metrics_history:
            latest_gpu = self.gpu_metrics_history[-1]['metrics']
            if latest_gpu:
                gpu_summary = {
                    'utilization_percent': latest_gpu[0].utilization_percent,
                    'memory_used_mb': latest_gpu[0].memory_used_mb,
                    'memory_percent': latest_gpu[0].memory_percent,
                    'temperature_c': latest_gpu[0].temperature_c,
                    'power_draw_w': latest_gpu[0].power_draw_w
                }
        
        # System metrics
        system_metrics = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage_percent': psutil.disk_usage('/').percent
        }
        
        return {
            'timestamp': current_time,
            'uptime_seconds': uptime,
            'jobs': {
                'total_completed': self.jobs_completed,
                'total_failed': self.jobs_failed,
                'active_jobs': len(self.active_jobs),
                'success_rate_percent': (self.jobs_completed / (self.jobs_completed + self.jobs_failed) * 100) if (self.jobs_completed + self.jobs_failed) > 0 else 0,
                'recent_hourly': {
                    'completed': recent_completed,
                    'failed': recent_failed
                }
            },
            'performance': {
                'avg_processing_time_seconds': avg_processing_time,
                'avg_frames_per_second': avg_frames_per_second,
                'avg_model_time_per_frame_ms': avg_model_time_per_frame * 1000 if avg_model_time_per_frame else 0,
                'total_frames_processed': self.total_frames_processed
            },
            'failures': dict(self.failure_reasons),
            'gpu': gpu_summary,
            'system': system_metrics
        }
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _get_gpu_info(self) -> List[GPUMetrics]:
        """Get detailed GPU information using nvidia-smi."""
        try:
            cmd = [
                'nvidia-smi',
                '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                raise Exception(f"nvidia-smi failed: {result.stderr}")
            
            gpu_metrics = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [part.strip() for part in line.split(',')]
                    if len(parts) >= 6:
                        gpu_id = int(parts[0])
                        utilization = float(parts[1])
                        memory_used = float(parts[2])
                        memory_total = float(parts[3])
                        temperature = float(parts[4])
                        power_draw = float(parts[5]) if parts[5] != '[Not Supported]' else 0.0
                        
                        memory_percent = (memory_used / memory_total * 100) if memory_total > 0 else 0
                        
                        gpu_metrics.append(GPUMetrics(
                            gpu_id=gpu_id,
                            utilization_percent=utilization,
                            memory_used_mb=memory_used,
                            memory_total_mb=memory_total,
                            memory_percent=memory_percent,
                            temperature_c=temperature,
                            power_draw_w=power_draw
                        ))
            
            return gpu_metrics
            
        except Exception as e:
            logger.error("Failed to get GPU info", error=str(e))
            return []


class FailureClassifier:
    """
    Classifies job failures into categories for better operational insights.
    """
    
    @staticmethod
    def classify_failure(error_message: str, exception_type: str = None) -> str:
        """
        Classify failure based on error message and exception type.
        
        Args:
            error_message: Error message from the failure
            exception_type: Type of exception if available
            
        Returns:
            Failure category string
        """
        error_lower = error_message.lower()
        
        # GPU/CUDA related errors
        if any(keyword in error_lower for keyword in ['cuda', 'gpu', 'out of memory', 'cudnn']):
            if 'out of memory' in error_lower:
                return 'gpu_oom'
            elif 'cuda' in error_lower:
                return 'cuda_error'
            else:
                return 'gpu_error'
        
        # Model loading errors
        if any(keyword in error_lower for keyword in ['model', 'checkpoint', 'weights']):
            return 'model_loading_error'
        
        # Video processing errors
        if any(keyword in error_lower for keyword in ['ffmpeg', 'video', 'codec', 'frame']):
            return 'video_processing_error'
        
        # Network/IO errors
        if any(keyword in error_lower for keyword in ['connection', 'network', 'timeout', 'io']):
            return 'network_io_error'
        
        # Storage errors
        if any(keyword in error_lower for keyword in ['disk', 'space', 'storage', 's3', 'bucket']):
            return 'storage_error'
        
        # Validation errors
        if any(keyword in error_lower for keyword in ['validation', 'invalid', 'format']):
            return 'validation_error'
        
        # Timeout errors
        if 'timeout' in error_lower:
            return 'timeout_error'
        
        # Default category
        return 'unknown_error'


# Global metrics collector instance
worker_metrics = WorkerMetricsCollector()


def start_job_metrics(job_id: str) -> JobMetrics:
    """Start tracking metrics for a job."""
    return worker_metrics.start_job(job_id)


def complete_job_metrics(job_id: str, frames_processed: int = 0, 
                        model_time_per_frame: float = None, success: bool = True,
                        error_message: str = None, exception_type: str = None):
    """Complete job metrics tracking."""
    failure_reason = None
    if not success and error_message:
        failure_reason = FailureClassifier.classify_failure(error_message, exception_type)
    
    worker_metrics.complete_job(
        job_id=job_id,
        frames_processed=frames_processed,
        model_time_per_frame=model_time_per_frame,
        success=success,
        failure_reason=failure_reason
    )


def get_worker_metrics() -> Dict[str, Any]:
    """Get current worker performance metrics."""
    return worker_metrics.get_performance_summary()


def get_gpu_metrics() -> List[GPUMetrics]:
    """Get current GPU metrics."""
    return worker_metrics.get_gpu_metrics()


async def start_gpu_monitoring(interval_seconds: int = 30):
    """Start background GPU monitoring task."""
    if not worker_metrics.gpu_available:
        logger.info("GPU monitoring not available")
        return
    
    logger.info("Starting GPU monitoring", interval_seconds=interval_seconds)
    
    while True:
        try:
            worker_metrics.record_gpu_metrics()
            await asyncio.sleep(interval_seconds)
        except Exception as e:
            logger.error("GPU monitoring error", error=str(e))
            await asyncio.sleep(interval_seconds)
