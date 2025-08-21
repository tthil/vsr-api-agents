#!/usr/bin/env python3
"""
Script to validate GPU availability and configuration for the VSR worker.
"""

import argparse
import json
import sys
import time
from typing import Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.utils.benchmark as benchmark
except ImportError:
    print("Error: PyTorch is not installed. Please install it with:")
    print("pip install torch torchvision")
    sys.exit(1)


def check_cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def get_cuda_info() -> Dict:
    """Get CUDA information."""
    if not check_cuda_available():
        return {"error": "CUDA not available"}
    
    return {
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": torch.cuda.get_device_capability(0),
        "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**2:.2f} MB",
        "memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**2:.2f} MB",
        "max_memory_allocated": f"{torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB",
        "max_memory_reserved": f"{torch.cuda.max_memory_reserved(0) / 1024**2:.2f} MB",
    }


def run_benchmark(device: str = "cuda", size: int = 2048, dtype: torch.dtype = torch.float32) -> Dict:
    """
    Run a simple matrix multiplication benchmark.
    
    Args:
        device: Device to run on ('cuda' or 'cpu')
        size: Matrix size
        dtype: Data type
        
    Returns:
        Dict with benchmark results
    """
    if device == "cuda" and not check_cuda_available():
        return {"error": "CUDA not available for benchmarking"}
    
    # Create random matrices
    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)
    
    # Warm up
    for _ in range(3):
        torch.matmul(a, b)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    t0 = time.time()
    for _ in range(5):
        c = torch.matmul(a, b)
        if device == "cuda":
            torch.cuda.synchronize()
    t1 = time.time()
    
    return {
        "device": device,
        "matrix_size": size,
        "dtype": str(dtype),
        "time_per_iteration_ms": (t1 - t0) * 1000 / 5,
    }


def test_simple_model(device: str = "cuda") -> Dict:
    """
    Test a simple CNN model.
    
    Args:
        device: Device to run on ('cuda' or 'cpu')
        
    Returns:
        Dict with test results
    """
    if device == "cuda" and not check_cuda_available():
        return {"error": "CUDA not available for model testing"}
    
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(2)
            self.fc = nn.Linear(32 * 56 * 56, 10)
        
        def forward(self, x):
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = x.view(-1, 32 * 56 * 56)
            x = self.fc(x)
            return x
    
    # Create model and move to device
    model = SimpleCNN().to(device)
    
    # Create random input
    x = torch.randn(1, 3, 224, 224, device=device)
    
    # Warm up
    for _ in range(3):
        model(x)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Test inference time
    t0 = time.time()
    for _ in range(10):
        model(x)
        if device == "cuda":
            torch.cuda.synchronize()
    t1 = time.time()
    
    return {
        "device": device,
        "model": "SimpleCNN",
        "input_shape": [1, 3, 224, 224],
        "time_per_inference_ms": (t1 - t0) * 1000 / 10,
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Validate GPU for VSR worker")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument("--model-test", action="store_true", help="Run model test")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    args = parser.parse_args()
    
    # Check CUDA availability
    cuda_available = check_cuda_available()
    
    results = {
        "cuda_available": cuda_available,
        "pytorch_version": torch.__version__,
    }
    
    if cuda_available:
        results["cuda_info"] = get_cuda_info()
        
        if args.benchmark:
            results["benchmark_cuda"] = run_benchmark(device="cuda")
            results["benchmark_cpu"] = run_benchmark(device="cpu")
        
        if args.model_test:
            results["model_test_cuda"] = test_simple_model(device="cuda")
            results["model_test_cpu"] = test_simple_model(device="cpu")
    
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print("\n=== GPU Validation Results ===\n")
        print(f"PyTorch Version: {results['pytorch_version']}")
        print(f"CUDA Available: {results['cuda_available']}")
        
        if cuda_available:
            cuda_info = results["cuda_info"]
            print("\n--- CUDA Information ---")
            print(f"CUDA Version: {cuda_info['cuda_version']}")
            print(f"Device Count: {cuda_info['device_count']}")
            print(f"Current Device: {cuda_info['current_device']}")
            print(f"Device Name: {cuda_info['device_name']}")
            print(f"Device Capability: {cuda_info['device_capability']}")
            print(f"Memory Allocated: {cuda_info['memory_allocated']}")
            print(f"Memory Reserved: {cuda_info['memory_reserved']}")
            
            if args.benchmark:
                print("\n--- Benchmark Results ---")
                cuda_bench = results["benchmark_cuda"]
                cpu_bench = results["benchmark_cpu"]
                print(f"CUDA: {cuda_bench['time_per_iteration_ms']:.2f} ms per iteration")
                print(f"CPU: {cpu_bench['time_per_iteration_ms']:.2f} ms per iteration")
                print(f"Speedup: {cpu_bench['time_per_iteration_ms'] / cuda_bench['time_per_iteration_ms']:.2f}x")
            
            if args.model_test:
                print("\n--- Model Test Results ---")
                cuda_test = results["model_test_cuda"]
                cpu_test = results["model_test_cpu"]
                print(f"CUDA: {cuda_test['time_per_inference_ms']:.2f} ms per inference")
                print(f"CPU: {cpu_test['time_per_inference_ms']:.2f} ms per inference")
                print(f"Speedup: {cpu_test['time_per_inference_ms'] / cuda_test['time_per_inference_ms']:.2f}x")
        else:
            print("\nWarning: CUDA is not available. Please check your GPU installation.")
            print("The VSR worker requires a CUDA-capable GPU to run efficiently.")


if __name__ == "__main__":
    main()
