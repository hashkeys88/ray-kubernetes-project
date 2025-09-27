# Ray on Kubernetes Project

This project demonstrates how to deploy and manage Ray clusters on Kubernetes for running distributed machine learning workloads at scale.

## Architecture Overview

The project implements a scalable Ray cluster architecture with the following components:

- **Ray Head Node**: Coordinates the cluster and manages job scheduling
- **Ray Worker Nodes**: Execute distributed tasks and computations
- **Kubernetes Integration**: Native K8s deployment with auto-scaling
- **Monitoring**: Prometheus and Grafana for observability
- **Storage**: Persistent volumes for data and model storage

## Project Structure

```
ray-kubernetes-project/
├── kubernetes/                 # Kubernetes manifests and configurations
│   ├── manifests/             # K8s deployment files
│   └── configs/               # Ray cluster configurations
├── ray-apps/                  # Sample Ray applications
│   ├── training/              # Distributed training examples
│   ├── hyperparameter-tuning/ # Hyperparameter optimization
│   └── distributed-computing/ # General distributed computing
├── monitoring/                # Monitoring and observability
├── scripts/                   # Deployment and utility scripts
└── docs/                      # Documentation
```

## Features

- **Auto-scaling**: Dynamic worker node scaling based on workload
- **Resource Management**: GPU and CPU resource allocation
- **Fault Tolerance**: Automatic recovery from node failures
- **Multi-tenancy**: Support for multiple teams and projects
- **Monitoring**: Comprehensive metrics and logging

## Quick Start

1. **Prerequisites**:
   - Kubernetes cluster (v1.20+)
   - kubectl configured
   - Docker for building images

2. **Deploy Ray Cluster**:
   ```bash
   ./scripts/deploy-ray-cluster.sh
   ```

3. **Run Sample Applications**:
   ```bash
   python ray-apps/training/distributed_training.py
   ```

## Components

### Ray Cluster Configuration
- Head node with job scheduling capabilities
- Worker nodes with auto-scaling
- Resource quotas and limits
- Network policies for security

### Sample Applications
- **Distributed Training**: Multi-node PyTorch training
- **Hyperparameter Tuning**: Ray Tune optimization
- **Distributed Computing**: General-purpose distributed tasks

### Monitoring Stack
- Prometheus for metrics collection
- Grafana for visualization
- Ray Dashboard integration
- Custom metrics and alerts

## Configuration

Key configuration files:
- `kubernetes/configs/ray-cluster.yaml`: Main cluster configuration
- `kubernetes/configs/ray-head.yaml`: Head node configuration
- `kubernetes/configs/ray-worker.yaml`: Worker node configuration

## Deployment

See [deployment guide](docs/deployment.md) for detailed instructions.

## Monitoring

Access the Ray dashboard at `http://ray-head-service:8265` after deployment.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details.
