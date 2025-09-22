#!/usr/bin/env python3
"""
Distributed Training Example with Ray
Inspired by Uber's Ray on Kubernetes setup
"""

import ray
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Ray
ray.init(address='ray://ray-head-service:10001', ignore_reinit_error=True)

class SimpleModel(nn.Module):
    """Simple neural network for demonstration"""
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

@ray.remote
class TrainingWorker:
    """Remote training worker for distributed training"""
    
    def __init__(self, worker_id: int, data_shard: torch.Tensor, labels_shard: torch.Tensor):
        self.worker_id = worker_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model and optimizer
        self.model = SimpleModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        # Create data loader
        dataset = TensorDataset(data_shard, labels_shard)
        self.data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        logger.info(f"Worker {worker_id} initialized on device {self.device}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch and return metrics"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(self.data_loader)
        
        return {
            'worker_id': self.worker_id,
            'loss': avg_loss,
            'accuracy': accuracy,
            'samples': total
        }
    
    def get_model_state(self) -> Dict[str, Any]:
        """Get current model state for aggregation"""
        return {
            'worker_id': self.worker_id,
            'state_dict': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict()
        }
    
    def set_model_state(self, state_dict: Dict[str, Any]):
        """Set model state from aggregated parameters"""
        self.model.load_state_dict(state_dict)
        logger.info(f"Worker {self.worker_id} updated with aggregated parameters")

def create_synthetic_data(num_samples: int = 10000, num_features: int = 784) -> tuple:
    """Create synthetic dataset for training"""
    # Generate random data
    X = torch.randn(num_samples, num_features)
    y = torch.randint(0, 10, (num_samples,))
    
    # Normalize data
    X = (X - X.mean()) / X.std()
    
    return X, y

def split_data_for_workers(data: torch.Tensor, labels: torch.Tensor, num_workers: int) -> list:
    """Split data among workers"""
    chunk_size = len(data) // num_workers
    data_shards = []
    label_shards = []
    
    for i in range(num_workers):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_workers - 1 else len(data)
        
        data_shards.append(data[start_idx:end_idx])
        label_shards.append(labels[start_idx:end_idx])
    
    return list(zip(data_shards, label_shards))

def aggregate_model_parameters(worker_states: list) -> Dict[str, Any]:
    """Aggregate model parameters from all workers"""
    if not worker_states:
        return {}
    
    # Get the first worker's state as base
    aggregated_state = {}
    first_state = worker_states[0]['state_dict']
    
    for key in first_state.keys():
        # Average parameters across all workers
        param_sum = torch.zeros_like(first_state[key])
        for worker_state in worker_states:
            param_sum += worker_state['state_dict'][key]
        
        aggregated_state[key] = param_sum / len(worker_states)
    
    return aggregated_state

def distributed_training(num_workers: int = 4, num_epochs: int = 10):
    """Main distributed training function"""
    logger.info(f"Starting distributed training with {num_workers} workers for {num_epochs} epochs")
    
    # Create synthetic data
    X, y = create_synthetic_data()
    logger.info(f"Created dataset with {len(X)} samples")
    
    # Split data among workers
    data_shards = split_data_for_workers(X, y, num_workers)
    logger.info(f"Split data into {len(data_shards)} shards")
    
    # Create workers
    workers = []
    for i, (data_shard, labels_shard) in enumerate(data_shards):
        worker = TrainingWorker.remote(i, data_shard, labels_shard)
        workers.append(worker)
    
    logger.info(f"Created {len(workers)} training workers")
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Train all workers in parallel
        training_futures = [worker.train_epoch.remote() for worker in workers]
        training_results = ray.get(training_futures)
        
        # Aggregate model parameters
        model_state_futures = [worker.get_model_state.remote() for worker in workers]
        model_states = ray.get(model_state_futures)
        aggregated_params = aggregate_model_parameters(model_states)
        
        # Update all workers with aggregated parameters
        if aggregated_params:
            update_futures = [worker.set_model_state.remote(aggregated_params) for worker in workers]
            ray.get(update_futures)
        
        # Calculate epoch metrics
        total_loss = sum(result['loss'] for result in training_results) / len(training_results)
        total_accuracy = sum(result['accuracy'] for result in training_results) / len(training_results)
        total_samples = sum(result['samples'] for result in training_results)
        
        epoch_time = time.time() - epoch_start_time
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                   f"Loss: {total_loss:.4f}, "
                   f"Accuracy: {total_accuracy:.2f}%, "
                   f"Samples: {total_samples}, "
                   f"Time: {epoch_time:.2f}s")
    
    logger.info("Distributed training completed successfully!")
    
    # Clean up
    ray.shutdown()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Distributed Training with Ray')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    
    args = parser.parse_args()
    
    try:
        distributed_training(num_workers=args.workers, num_epochs=args.epochs)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        ray.shutdown()
        raise
