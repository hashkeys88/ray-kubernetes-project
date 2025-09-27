#!/usr/bin/env python3
"""
Ray Tune Hyperparameter Optimization Example
Demonstrates distributed hyperparameter optimization on Ray clusters
"""

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TuneModel(nn.Module):
    """Model for hyperparameter tuning"""
    def __init__(self, config: Dict[str, Any]):
        super(TuneModel, self).__init__()
        
        self.hidden_size = config.get("hidden_size", 128)
        self.dropout_rate = config.get("dropout_rate", 0.2)
        self.learning_rate = config.get("learning_rate", 0.001)
        
        self.fc1 = nn.Linear(784, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_rate)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def create_tune_dataset():
    """Create dataset for hyperparameter tuning"""
    # Generate synthetic data
    X = torch.randn(5000, 784)
    y = torch.randint(0, 10, (5000,))
    
    # Normalize
    X = (X - X.mean()) / X.std()
    
    # Split into train/val
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    return X_train, y_train, X_val, y_val

def train_model(config: Dict[str, Any], checkpoint_dir=None):
    """Training function for Ray Tune"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = TuneModel(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
    # Create data loaders
    X_train, y_train, X_val, y_val = create_tune_dataset()
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=config.get("batch_size", 32), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.get("batch_size", 32), shuffle=False)
    
    # Training loop
    for epoch in range(config.get("epochs", 10)):
        # Training
        model.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        val_accuracy = 100. * correct / total
        avg_val_loss = val_loss / len(val_loader)
        
        # Report metrics to Ray Tune
        tune.report(
            val_loss=avg_val_loss,
            val_accuracy=val_accuracy,
            train_loss=train_loss / len(train_loader),
            epoch=epoch
        )
        
        # Save checkpoint
        if checkpoint_dir:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_accuracy': val_accuracy,
                'config': config
            }, f"{checkpoint_dir}/checkpoint_{epoch}.pth")

def run_hyperparameter_tuning():
    """Run hyperparameter optimization with Ray Tune"""
    logger.info("Starting hyperparameter optimization with Ray Tune")
    
    # Initialize Ray
    ray.init(address='ray://ray-head-service:10001', ignore_reinit_error=True)
    
    # Define search space
    search_space = {
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "hidden_size": tune.choice([64, 128, 256, 512]),
        "dropout_rate": tune.uniform(0.1, 0.5),
        "batch_size": tune.choice([16, 32, 64, 128]),
        "epochs": 10
    }
    
    # Define scheduler
    scheduler = ASHAScheduler(
        metric="val_accuracy",
        mode="max",
        max_t=10,
        grace_period=3,
        reduction_factor=2
    )
    
    # Define search algorithm
    search_alg = OptunaSearch(metric="val_accuracy", mode="max")
    
    # Run tuning
    analysis = tune.run(
        train_model,
        config=search_space,
        num_samples=20,
        scheduler=scheduler,
        search_alg=search_alg,
        checkpoint_at_end=True,
        local_dir="./ray_tune_results",
        name="hyperparameter_tuning",
        resources_per_trial={"cpu": 1, "gpu": 0},
        verbose=1
    )
    
    # Get best results
    best_trial = analysis.get_best_trial("val_accuracy", "max", "last")
    best_config = best_trial.config
    best_metrics = best_trial.last_result
    
    logger.info("Hyperparameter tuning completed!")
    logger.info(f"Best config: {best_config}")
    logger.info(f"Best metrics: {best_metrics}")
    
    # Save results
    results_df = analysis.results_df
    results_df.to_csv("hyperparameter_tuning_results.csv", index=False)
    logger.info("Results saved to hyperparameter_tuning_results.csv")
    
    return analysis

def run_parallel_tuning():
    """Run multiple tuning experiments in parallel"""
    logger.info("Starting parallel hyperparameter tuning experiments")
    
    # Initialize Ray
    ray.init(address='ray://ray-head-service:10001', ignore_reinit_error=True)
    
    # Define different search spaces for different experiments
    experiments = {
        "experiment_1": {
            "learning_rate": tune.loguniform(1e-4, 1e-2),
            "hidden_size": tune.choice([64, 128]),
            "dropout_rate": tune.uniform(0.1, 0.3),
            "batch_size": tune.choice([32, 64]),
            "epochs": 5
        },
        "experiment_2": {
            "learning_rate": tune.loguniform(1e-3, 1e-1),
            "hidden_size": tune.choice([256, 512]),
            "dropout_rate": tune.uniform(0.2, 0.5),
            "batch_size": tune.choice([16, 32]),
            "epochs": 8
        }
    }
    
    # Run experiments in parallel
    analyses = {}
    for exp_name, search_space in experiments.items():
        logger.info(f"Starting experiment: {exp_name}")
        
        analysis = tune.run(
            train_model,
            config=search_space,
            num_samples=10,
            local_dir=f"./ray_tune_results/{exp_name}",
            name=exp_name,
            resources_per_trial={"cpu": 1, "gpu": 0},
            verbose=1
        )
        
        analyses[exp_name] = analysis
        
        # Get best results for this experiment
        best_trial = analysis.get_best_trial("val_accuracy", "max", "last")
        logger.info(f"{exp_name} - Best accuracy: {best_trial.last_result['val_accuracy']:.2f}%")
    
    logger.info("All parallel experiments completed!")
    return analyses

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Ray Tune Hyperparameter Optimization')
    parser.add_argument('--mode', choices=['single', 'parallel'], default='single',
                       help='Run single or parallel tuning experiments')
    parser.add_argument('--samples', type=int, default=20,
                       help='Number of samples for tuning')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'single':
            analysis = run_hyperparameter_tuning()
        else:
            analyses = run_parallel_tuning()
    except Exception as e:
        logger.error(f"Hyperparameter tuning failed: {e}")
        ray.shutdown()
        raise
    finally:
        ray.shutdown()
