# Ray on Kubernetes Deployment Guide

This guide provides detailed instructions for deploying and managing Ray clusters on Kubernetes, inspired by Uber's journey to Ray on Kubernetes.

## Prerequisites

### System Requirements
- Kubernetes cluster (v1.20 or higher)
- kubectl configured and connected to your cluster
- Docker (for building custom images)
- At least 4 CPU cores and 8GB RAM available in the cluster
- Storage class configured for persistent volumes

### Software Requirements
- kubectl v1.20+
- Docker (optional, for custom images)
- Python 3.8+ (for running examples)

## Quick Start

### 1. Clone and Setup
```bash
git clone <repository-url>
cd ray-kubernetes-project
```

### 2. Deploy Ray Cluster
```bash
./scripts/deploy-ray-cluster.sh deploy
```

### 3. Verify Deployment
```bash
./scripts/deploy-ray-cluster.sh status
```

### 4. Run Examples
```bash
./scripts/run-examples.sh all
```

## Detailed Deployment

### Step 1: Create Namespace and RBAC
The deployment script automatically creates:
- `ray-system` namespace
- Service account with appropriate permissions
- Cluster role and role binding for Ray operations

### Step 2: Deploy Storage
Persistent storage is configured for:
- Ray data and logs
- Prometheus metrics storage
- Grafana dashboard storage

### Step 3: Deploy Ray Cluster
The Ray cluster consists of:
- **Head Node**: Coordinates the cluster and manages job scheduling
- **Worker Nodes**: Execute distributed tasks (auto-scaling enabled)
- **Services**: Expose Ray services within the cluster

### Step 4: Deploy Monitoring
Monitoring stack includes:
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboards
- **Custom Metrics**: Ray-specific metrics and health checks

## Configuration

### Ray Cluster Configuration
Key configuration files:
- `kubernetes/configs/ray-cluster.yaml`: Main cluster settings
- `kubernetes/manifests/ray-head-deployment.yaml`: Head node configuration
- `kubernetes/manifests/ray-worker-deployment.yaml`: Worker node configuration

### Resource Allocation
Default resource allocation:
- Head node: 2 CPU, 4GB RAM
- Worker nodes: 2 CPU, 2GB RAM each
- Auto-scaling: 1-10 worker nodes based on CPU/memory usage

### Customization
To customize the deployment:

1. **Modify resource limits** in deployment files
2. **Adjust auto-scaling** parameters in HPA configuration
3. **Update Ray configuration** in ConfigMap
4. **Add custom images** by building and pushing to your registry

## Accessing Services

### Ray Dashboard
```bash
kubectl port-forward -n ray-system service/ray-head-service 8265:8265
```
Open: http://localhost:8265

### Prometheus
```bash
kubectl port-forward -n ray-system service/prometheus-service 9090:9090
```
Open: http://localhost:9090

### Grafana
```bash
kubectl port-forward -n ray-system service/grafana-service 3000:3000
```
Open: http://localhost:3000 (admin/admin123)

## Running Applications

### Distributed Training
```bash
./scripts/run-examples.sh training
```

### Hyperparameter Tuning
```bash
./scripts/run-examples.sh tuning
```

### Ray Serve
```bash
./scripts/run-examples.sh serve
```

### Custom Applications
```bash
./scripts/run-examples.sh custom /path/to/your/script.py
```

## Monitoring and Observability

### Key Metrics
- **Cluster Health**: Node status, resource utilization
- **Task Performance**: Execution time, success rate
- **Resource Usage**: CPU, memory, GPU utilization
- **Custom Metrics**: Application-specific metrics

### Alerts
Pre-configured alerts for:
- Node failures
- High resource usage
- Task failures
- Service unavailability

### Dashboards
- **Ray Cluster Overview**: Node status, resource usage
- **Task Monitoring**: Task execution and performance
- **Resource Utilization**: CPU, memory, storage usage
- **Custom Dashboards**: Application-specific visualizations

## Troubleshooting

### Common Issues

#### Ray Head Node Not Starting
```bash
kubectl logs -n ray-system deployment/ray-head
kubectl describe pod -n ray-system -l app=ray-head
```

#### Worker Nodes Not Connecting
```bash
kubectl logs -n ray-system deployment/ray-worker
kubectl get pods -n ray-system -l app=ray-worker
```

#### Resource Issues
```bash
kubectl top pods -n ray-system
kubectl describe nodes
```

### Debugging Commands
```bash
# Check cluster status
kubectl get pods -n ray-system
kubectl get services -n ray-system

# View logs
kubectl logs -n ray-system deployment/ray-head
kubectl logs -n ray-system deployment/ray-worker

# Check resource usage
kubectl top pods -n ray-system
kubectl top nodes

# Access Ray cluster
kubectl exec -n ray-system deployment/ray-head -- ray status
```

## Scaling

### Manual Scaling
```bash
kubectl scale deployment ray-worker -n ray-system --replicas=5
```

### Auto-scaling
The cluster is configured with Horizontal Pod Autoscaler (HPA):
- Scales based on CPU usage (70% threshold)
- Scales based on memory usage (80% threshold)
- Minimum: 1 worker node
- Maximum: 10 worker nodes

### Custom Scaling
To modify auto-scaling behavior:
1. Edit `kubernetes/manifests/ray-worker-deployment.yaml`
2. Update HPA configuration
3. Apply changes: `kubectl apply -f kubernetes/manifests/ray-worker-deployment.yaml`

## Security

### Network Policies
- Ray services are isolated within the namespace
- External access is controlled via port-forwarding
- Inter-pod communication is restricted to necessary ports

### RBAC
- Service account with minimal required permissions
- Cluster role with specific resource access
- Role binding scoped to the ray-system namespace

### Best Practices
- Use secrets for sensitive configuration
- Enable network policies for production
- Regular security updates for images
- Monitor access logs and metrics

## Production Considerations

### High Availability
- Deploy multiple head nodes (with external coordination)
- Use persistent storage for critical data
- Configure proper resource limits and requests
- Implement health checks and monitoring

### Performance Optimization
- Use GPU nodes for ML workloads
- Optimize network configuration
- Tune Ray parameters for your workload
- Monitor and adjust resource allocation

### Backup and Recovery
- Regular backups of persistent volumes
- Configuration management with Git
- Disaster recovery procedures
- Data replication strategies

## Cleanup

### Remove Ray Cluster
```bash
./scripts/deploy-ray-cluster.sh cleanup
```

### Manual Cleanup
```bash
kubectl delete namespace ray-system
```

## Support and Resources

### Documentation
- [Ray Documentation](https://docs.ray.io/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)

### Community
- [Ray Slack](https://ray-distributed.slack.com/)
- [Kubernetes Slack](https://kubernetes.slack.com/)
- [GitHub Issues](https://github.com/ray-project/ray/issues)

### Professional Support
- Ray Enterprise support
- Kubernetes managed services
- Cloud provider support
