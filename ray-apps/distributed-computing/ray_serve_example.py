#!/usr/bin/env python3
"""
Ray Serve Example for Model Serving
Demonstrates distributed model serving on Ray clusters
"""

import ray
from ray import serve
import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Any, List
import asyncio
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleModel(nn.Module):
    """Simple model for serving"""
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

@serve.deployment
class ModelDeployment:
    """Ray Serve deployment for model inference"""
    
    def __init__(self):
        self.model = SimpleModel()
        self.model.eval()
        logger.info("Model deployment initialized")
    
    async def __call__(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle inference requests"""
        try:
            # Extract input data
            input_data = request.get("data", [])
            if not input_data:
                return {"error": "No input data provided"}
            
            # Convert to tensor
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
            
            # Run inference
            with torch.no_grad():
                output = self.model(input_tensor)
                predictions = torch.softmax(output, dim=1)
                predicted_classes = torch.argmax(predictions, dim=1)
            
            return {
                "predictions": predicted_classes.tolist(),
                "probabilities": predictions.tolist(),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {"error": str(e), "status": "failed"}

@serve.deployment
class BatchProcessor:
    """Batch processing deployment"""
    
    def __init__(self):
        self.processed_batches = 0
        logger.info("Batch processor initialized")
    
    async def __call__(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process batch requests"""
        try:
            batch_data = request.get("batch", [])
            batch_size = len(batch_data)
            
            # Simulate batch processing
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # Process each item in the batch
            results = []
            for item in batch_data:
                # Simulate some processing
                processed_item = {
                    "id": item.get("id", "unknown"),
                    "processed": True,
                    "timestamp": time.time(),
                    "result": f"processed_{item.get('id', 'unknown')}"
                }
                results.append(processed_item)
            
            self.processed_batches += 1
            
            return {
                "batch_id": request.get("batch_id", "unknown"),
                "batch_size": batch_size,
                "results": results,
                "processed_batches": self.processed_batches,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            return {"error": str(e), "status": "failed"}

@serve.deployment
class HealthChecker:
    """Health check deployment"""
    
    def __init__(self):
        self.start_time = time.time()
        logger.info("Health checker initialized")
    
    async def __call__(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Health check endpoint"""
        uptime = time.time() - self.start_time
        
        return {
            "status": "healthy",
            "uptime": uptime,
            "timestamp": time.time(),
            "version": "1.0.0"
        }

def create_sample_data(num_samples: int = 100) -> List[Dict[str, Any]]:
    """Create sample data for testing"""
    data = []
    for i in range(num_samples):
        # Generate random input data (784 features)
        input_data = np.random.randn(784).tolist()
        
        data.append({
            "id": f"sample_{i}",
            "data": input_data,
            "metadata": {
                "created_at": time.time(),
                "source": "synthetic"
            }
        })
    
    return data

async def test_model_serving():
    """Test model serving functionality"""
    logger.info("Testing model serving...")
    
    # Create sample data
    sample_data = create_sample_data(10)
    
    # Test model deployment
    model_handle = serve.get_deployment("model_deployment").get_handle()
    
    for i, sample in enumerate(sample_data):
        request = {"data": [sample["data"]]}
        response = await model_handle.remote(request)
        logger.info(f"Sample {i}: {response}")
    
    logger.info("Model serving test completed")

async def test_batch_processing():
    """Test batch processing functionality"""
    logger.info("Testing batch processing...")
    
    # Create batch data
    batch_data = []
    for i in range(50):
        batch_data.append({
            "id": f"batch_item_{i}",
            "data": f"data_{i}",
            "priority": i % 3
        })
    
    # Test batch processor
    batch_handle = serve.get_deployment("batch_processor").get_handle()
    
    request = {
        "batch_id": "test_batch_001",
        "batch": batch_data
    }
    
    response = await batch_handle.remote(request)
    logger.info(f"Batch processing result: {response}")
    
    logger.info("Batch processing test completed")

async def test_health_check():
    """Test health check functionality"""
    logger.info("Testing health check...")
    
    health_handle = serve.get_deployment("health_checker").get_handle()
    response = await health_handle.remote({})
    logger.info(f"Health check result: {response}")
    
    logger.info("Health check test completed")

def setup_ray_serve():
    """Setup Ray Serve with deployments"""
    logger.info("Setting up Ray Serve...")
    
    # Initialize Ray
    ray.init(address='ray://ray-head-service:10001', ignore_reinit_error=True)
    
    # Start Ray Serve
    serve.start(detached=True, http_options={"host": "0.0.0.0", "port": 8000})
    
    # Deploy services
    ModelDeployment.deploy(name="model_deployment")
    BatchProcessor.deploy(name="batch_processor")
    HealthChecker.deploy(name="health_checker")
    
    logger.info("Ray Serve setup completed")
    logger.info("Services deployed:")
    logger.info("- Model Deployment: http://ray-head-service:8000/model_deployment")
    logger.info("- Batch Processor: http://ray-head-service:8000/batch_processor")
    logger.info("- Health Checker: http://ray-head-service:8000/health_checker")

async def run_serve_example():
    """Run the complete Ray Serve example"""
    try:
        # Setup Ray Serve
        setup_ray_serve()
        
        # Wait for deployments to be ready
        await asyncio.sleep(2)
        
        # Run tests
        await test_health_check()
        await test_model_serving()
        await test_batch_processing()
        
        logger.info("All Ray Serve tests completed successfully!")
        
        # Keep the service running
        logger.info("Ray Serve is running. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down Ray Serve...")
        serve.shutdown()
        ray.shutdown()
    except Exception as e:
        logger.error(f"Ray Serve example failed: {e}")
        serve.shutdown()
        ray.shutdown()
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Ray Serve Example')
    parser.add_argument('--mode', choices=['setup', 'test'], default='setup',
                       help='Setup Ray Serve or run tests')
    
    args = parser.parse_args()
    
    if args.mode == 'setup':
        setup_ray_serve()
    else:
        asyncio.run(run_serve_example())
