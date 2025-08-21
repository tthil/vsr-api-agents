"""
Comprehensive test suite for dual-mode worker architecture.

Tests both CPU and GPU modes with environment detection, model loading,
and processing pipeline validation.
"""

import pytest
import asyncio
import os
import tempfile
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from vsr_worker.gpu.environment import WorkerEnvironment
from vsr_worker.models.adaptive import AdaptiveModelFactory, AdaptiveModelInterface
from vsr_worker.config.dual_mode import (
    get_config_manager, 
    WorkerMode, 
    ModelQuality,
    validate_environment
)
from vsr_worker.dual_mode_consumer import DualModeVideoJobConsumer


class TestWorkerEnvironment:
    """Test worker environment detection and configuration."""
    
    @pytest.fixture
    def mock_env_vars(self):
        """Mock environment variables for testing."""
        with patch.dict(os.environ, {
            'WORKER_MODE': 'auto',
            'ENVIRONMENT': 'development',
            'MODEL_QUALITY': 'lightweight',
            'CPU_THREADS': '4',
            'MEMORY_LIMIT_MB': '2048'
        }):
            yield
    
    def test_auto_mode_detection_no_cuda(self, mock_env_vars):
        """Test automatic mode detection when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            env = WorkerEnvironment()
            assert env.mode == "cpu"
            assert env.device.type == "cpu"
    
    def test_auto_mode_detection_with_cuda(self, mock_env_vars):
        """Test automatic mode detection when CUDA is available."""
        with patch('torch.cuda.is_available', return_value=True):
            env = WorkerEnvironment()
            assert env.mode == "gpu"
            assert env.device.type == "cuda"
    
    def test_forced_cpu_mode(self):
        """Test forcing CPU mode via environment variable."""
        with patch.dict(os.environ, {'WORKER_MODE': 'cpu'}):
            env = WorkerEnvironment()
            assert env.mode == "cpu"
    
    def test_forced_gpu_mode_no_cuda(self):
        """Test forcing GPU mode when CUDA is not available."""
        with patch.dict(os.environ, {'WORKER_MODE': 'gpu'}):
            with patch('torch.cuda.is_available', return_value=False):
                with pytest.raises(RuntimeError, match="CUDA not available"):
                    WorkerEnvironment()
    
    @pytest.mark.asyncio
    async def test_environment_initialization(self, mock_env_vars):
        """Test environment initialization and info gathering."""
        with patch('torch.cuda.is_available', return_value=False):
            env = WorkerEnvironment()
            info = await env.initialize()
            
            assert "mode" in info
            assert "device" in info
            assert "cpu_info" in info
            assert info["mode"] == "cpu"


class TestConfigurationManager:
    """Test configuration management system."""
    
    def test_config_manager_initialization(self):
        """Test configuration manager initialization."""
        with patch.dict(os.environ, {
            'WORKER_MODE': 'cpu',
            'MODEL_QUALITY': 'lightweight'
        }):
            config_manager = get_config_manager()
            assert config_manager.config.worker.mode == WorkerMode.CPU
            assert config_manager.config.model.quality == ModelQuality.LIGHTWEIGHT
    
    def test_processing_config_cpu(self):
        """Test processing configuration for CPU mode."""
        with patch.dict(os.environ, {
            'WORKER_MODE': 'cpu',
            'PROCESSING_TIMEOUT': '300',
            'MAX_VIDEO_DURATION': '30'
        }):
            config_manager = get_config_manager()
            processing_config = config_manager.get_processing_config(WorkerMode.CPU)
            
            assert processing_config.timeout_seconds == 300
            assert processing_config.max_video_duration == 30
            assert processing_config.max_resolution == (1280, 720)
    
    def test_processing_config_gpu(self):
        """Test processing configuration for GPU mode."""
        with patch.dict(os.environ, {
            'WORKER_MODE': 'gpu',
            'PROCESSING_TIMEOUT': '60',
            'MAX_VIDEO_DURATION': '60'
        }):
            config_manager = get_config_manager()
            processing_config = config_manager.get_processing_config(WorkerMode.GPU)
            
            assert processing_config.timeout_seconds == 60
            assert processing_config.max_video_duration == 60
            assert processing_config.max_resolution == (1920, 1080)
    
    def test_environment_validation(self):
        """Test environment validation."""
        with patch.dict(os.environ, {
            'WORKER_MODE': 'cpu',
            'ENVIRONMENT': 'development',
            'MODEL_QUALITY': 'lightweight'
        }):
            assert validate_environment() is True
    
    def test_environment_validation_invalid(self):
        """Test environment validation with invalid configuration."""
        with patch.dict(os.environ, {
            'WORKER_MODE': 'invalid_mode'
        }):
            assert validate_environment() is False


class TestAdaptiveModels:
    """Test adaptive model interface and implementations."""
    
    @pytest.fixture
    def mock_worker_env(self):
        """Mock worker environment for testing."""
        env = Mock()
        env.mode = "cpu"
        env.device = Mock()
        env.device.type = "cpu"
        return env
    
    @pytest.mark.asyncio
    async def test_model_factory_cpu_mode(self, mock_worker_env):
        """Test model factory in CPU mode."""
        factory = AdaptiveModelFactory(mock_worker_env)
        
        # Test STTN model creation
        with patch('vsr_worker.models.adaptive.AdaptiveSTTNModel') as mock_sttn:
            mock_model = AsyncMock()
            mock_sttn.return_value = mock_model
            
            model = await factory.create_model("sttn")
            assert model is not None
            mock_sttn.assert_called_once_with(mock_worker_env)
    
    @pytest.mark.asyncio
    async def test_model_factory_unsupported_model(self, mock_worker_env):
        """Test model factory with unsupported model type."""
        factory = AdaptiveModelFactory(mock_worker_env)
        
        with pytest.raises(ValueError, match="Unsupported model type"):
            await factory.create_model("unsupported_model")
    
    @pytest.mark.asyncio
    async def test_model_lifecycle(self, mock_worker_env):
        """Test model loading and unloading lifecycle."""
        factory = AdaptiveModelFactory(mock_worker_env)
        
        with patch('vsr_worker.models.adaptive.AdaptiveSTTNModel') as mock_sttn:
            mock_model = AsyncMock()
            mock_sttn.return_value = mock_model
            
            # Create and load model
            model = await factory.create_model("sttn")
            mock_model.load_model.assert_called_once()
            
            # Unload all models
            await factory.unload_all_models()
            mock_model.unload_model.assert_called_once()


class TestDualModeConsumer:
    """Test dual-mode consumer implementation."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock all external dependencies."""
        with patch('vsr_worker.dual_mode_consumer.RabbitMQClient') as mock_rabbitmq, \
             patch('vsr_worker.dual_mode_consumer.get_db') as mock_db, \
             patch('vsr_worker.dual_mode_consumer.get_spaces_client') as mock_spaces, \
             patch('vsr_worker.dual_mode_consumer.notify_job_completion') as mock_webhook:
            
            mock_db_instance = AsyncMock()
            mock_db.return_value = mock_db_instance
            
            yield {
                'rabbitmq': mock_rabbitmq,
                'db': mock_db_instance,
                'spaces': mock_spaces,
                'webhook': mock_webhook
            }
    
    @pytest.mark.asyncio
    async def test_consumer_initialization(self, mock_dependencies):
        """Test consumer initialization."""
        with patch.dict(os.environ, {
            'WORKER_MODE': 'cpu',
            'ENVIRONMENT': 'development'
        }):
            consumer = DualModeVideoJobConsumer()
            
            with patch.object(consumer, 'worker_env') as mock_env:
                mock_env.initialize.return_value = {"mode": "cpu", "device": "cpu"}
                
                success = await consumer.initialize()
                assert success is True
    
    @pytest.mark.asyncio
    async def test_job_processing_success(self, mock_dependencies):
        """Test successful job processing."""
        consumer = DualModeVideoJobConsumer()
        
        # Mock worker environment
        mock_env = Mock()
        mock_env.mode = "cpu"
        consumer.worker_env = mock_env
        
        # Mock model factory
        mock_factory = AsyncMock()
        mock_model = AsyncMock()
        mock_factory.create_model.return_value = mock_model
        consumer.model_factory = mock_factory
        
        # Mock config manager
        mock_config = Mock()
        mock_config.get_processing_config.return_value = Mock(timeout_seconds=300)
        consumer.config_manager = mock_config
        
        job_data = {
            "job_id": "test-job-123",
            "mode": "sttn",
            "video_key": "uploads/test.mp4",
            "subtitle_area": {"x": 0, "y": 0, "width": 100, "height": 50}
        }
        
        # Mock database operations
        mock_dependencies['db'].jobs.update_one = AsyncMock()
        mock_dependencies['db'].jobs.find_one = AsyncMock(return_value={
            "_id": "test-job-123",
            "status": "completed"
        })
        mock_dependencies['db'].job_events.insert_one = AsyncMock()
        
        # Mock video processing
        with patch.object(consumer, '_process_video') as mock_process:
            mock_process.return_value = {
                "processed_video_key": "processed/test-job-123.mp4",
                "quality_metrics": {"psnr": 28.5}
            }
            
            success = await consumer.process_job(job_data)
            assert success is True
            
            # Verify model was created
            mock_factory.create_model.assert_called_once_with("sttn")
            
            # Verify database updates
            assert mock_dependencies['db'].jobs.update_one.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_job_processing_failure(self, mock_dependencies):
        """Test job processing failure handling."""
        consumer = DualModeVideoJobConsumer()
        
        # Mock worker environment
        mock_env = Mock()
        mock_env.mode = "cpu"
        consumer.worker_env = mock_env
        
        # Mock model factory to raise exception
        mock_factory = AsyncMock()
        mock_factory.create_model.side_effect = Exception("Model loading failed")
        consumer.model_factory = mock_factory
        
        job_data = {
            "job_id": "test-job-456",
            "mode": "sttn"
        }
        
        # Mock database operations
        mock_dependencies['db'].jobs.update_one = AsyncMock()
        mock_dependencies['db'].job_events.insert_one = AsyncMock()
        
        success = await consumer.process_job(job_data)
        assert success is False
        
        # Verify failure was recorded
        update_calls = mock_dependencies['db'].jobs.update_one.call_args_list
        failure_update = next(
            call for call in update_calls 
            if "FAILED" in str(call)
        )
        assert failure_update is not None


class TestIntegration:
    """Integration tests for the complete dual-mode system."""
    
    @pytest.mark.asyncio
    async def test_cpu_mode_end_to_end(self):
        """Test complete CPU mode workflow."""
        with patch.dict(os.environ, {
            'WORKER_MODE': 'cpu',
            'ENVIRONMENT': 'development',
            'MODEL_QUALITY': 'lightweight'
        }):
            # Test environment detection
            with patch('torch.cuda.is_available', return_value=False):
                env = WorkerEnvironment()
                assert env.mode == "cpu"
            
            # Test configuration
            config_manager = get_config_manager()
            assert config_manager.config.worker.mode == WorkerMode.CPU
            
            # Test model factory
            factory = AdaptiveModelFactory(env)
            assert factory.worker_env.mode == "cpu"
    
    @pytest.mark.asyncio
    async def test_gpu_mode_end_to_end(self):
        """Test complete GPU mode workflow."""
        with patch.dict(os.environ, {
            'WORKER_MODE': 'gpu',
            'ENVIRONMENT': 'production',
            'MODEL_QUALITY': 'high'
        }):
            # Test environment detection
            with patch('torch.cuda.is_available', return_value=True):
                env = WorkerEnvironment()
                assert env.mode == "gpu"
            
            # Test configuration
            config_manager = get_config_manager()
            assert config_manager.config.worker.mode == WorkerMode.GPU
            
            # Test model factory
            factory = AdaptiveModelFactory(env)
            assert factory.worker_env.mode == "gpu"


# Performance and stress tests
class TestPerformance:
    """Performance tests for dual-mode architecture."""
    
    @pytest.mark.asyncio
    async def test_model_loading_performance(self):
        """Test model loading performance in different modes."""
        with patch('torch.cuda.is_available', return_value=False):
            env = WorkerEnvironment()
            factory = AdaptiveModelFactory(env)
            
            # Mock model creation to measure timing
            with patch('vsr_worker.models.adaptive.AdaptiveSTTNModel') as mock_sttn:
                mock_model = AsyncMock()
                mock_sttn.return_value = mock_model
                
                import time
                start_time = time.time()
                await factory.create_model("sttn")
                load_time = time.time() - start_time
                
                # CPU mode should load quickly (mocked)
                assert load_time < 1.0
    
    @pytest.mark.asyncio
    async def test_concurrent_job_processing(self):
        """Test processing multiple jobs concurrently."""
        # This would test the system's ability to handle multiple
        # concurrent jobs in different modes
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
