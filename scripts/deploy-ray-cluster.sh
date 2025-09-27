#!/bin/bash

# Ray on Kubernetes Deployment Script
# Complete deployment script for Ray clusters on Kubernetes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="ray-system"
RAY_VERSION="2.8.0"
PROMETHEUS_VERSION="v2.45.0"
GRAFANA_VERSION="10.0.0"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed. Please install kubectl first."
        exit 1
    fi
    
    # Check if kubectl can connect to cluster
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
        exit 1
    fi
    
    # Check if Docker is installed (for building images)
    if ! command -v docker &> /dev/null; then
        log_warning "Docker is not installed. You may need it for building custom images."
    fi
    
    log_success "Prerequisites check completed"
}

create_namespace() {
    log_info "Creating namespace: $NAMESPACE"
    
    if kubectl get namespace $NAMESPACE &> /dev/null; then
        log_warning "Namespace $NAMESPACE already exists"
    else
        kubectl apply -f kubernetes/manifests/ray-namespace.yaml
        log_success "Namespace $NAMESPACE created"
    fi
}

deploy_storage() {
    log_info "Deploying storage components..."
    
    kubectl apply -f kubernetes/manifests/ray-storage.yaml
    log_success "Storage components deployed"
}

deploy_ray_cluster() {
    log_info "Deploying Ray cluster..."
    
    # Deploy Ray head node
    kubectl apply -f kubernetes/manifests/ray-head-deployment.yaml
    log_success "Ray head node deployed"
    
    # Wait for head node to be ready
    log_info "Waiting for Ray head node to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/ray-head -n $NAMESPACE
    
    # Deploy Ray worker nodes
    kubectl apply -f kubernetes/manifests/ray-worker-deployment.yaml
    log_success "Ray worker nodes deployed"
    
    # Wait for worker nodes to be ready
    log_info "Waiting for Ray worker nodes to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/ray-worker -n $NAMESPACE
}

deploy_monitoring() {
    log_info "Deploying monitoring stack..."
    
    # Deploy Prometheus
    kubectl apply -f monitoring/prometheus-deployment.yaml
    log_success "Prometheus deployed"
    
    # Deploy Grafana
    kubectl apply -f monitoring/grafana-deployment.yaml
    log_success "Grafana deployed"
    
    # Wait for monitoring components
    log_info "Waiting for monitoring components to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/prometheus -n $NAMESPACE
    kubectl wait --for=condition=available --timeout=300s deployment/grafana -n $NAMESPACE
}

verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check Ray head node
    if kubectl get pods -l app=ray-head -n $NAMESPACE | grep -q "Running"; then
        log_success "Ray head node is running"
    else
        log_error "Ray head node is not running"
        return 1
    fi
    
    # Check Ray worker nodes
    if kubectl get pods -l app=ray-worker -n $NAMESPACE | grep -q "Running"; then
        log_success "Ray worker nodes are running"
    else
        log_error "Ray worker nodes are not running"
        return 1
    fi
    
    # Check monitoring
    if kubectl get pods -l app=prometheus -n $NAMESPACE | grep -q "Running"; then
        log_success "Prometheus is running"
    else
        log_warning "Prometheus is not running"
    fi
    
    if kubectl get pods -l app=grafana -n $NAMESPACE | grep -q "Running"; then
        log_success "Grafana is running"
    else
        log_warning "Grafana is not running"
    fi
}

show_access_info() {
    log_info "Deployment completed! Access information:"
    echo ""
    echo "Ray Dashboard:"
    echo "  kubectl port-forward -n $NAMESPACE service/ray-head-service 8265:8265"
    echo "  Then open: http://localhost:8265"
    echo ""
    echo "Prometheus:"
    echo "  kubectl port-forward -n $NAMESPACE service/prometheus-service 9090:9090"
    echo "  Then open: http://localhost:9090"
    echo ""
    echo "Grafana:"
    echo "  kubectl port-forward -n $NAMESPACE service/grafana-service 3000:3000"
    echo "  Then open: http://localhost:3000 (admin/admin123)"
    echo ""
    echo "Ray Cluster Address:"
    echo "  ray://ray-head-service:10001"
    echo ""
}

cleanup() {
    log_info "Cleaning up Ray cluster..."
    
    kubectl delete -f kubernetes/manifests/ray-worker-deployment.yaml --ignore-not-found=true
    kubectl delete -f kubernetes/manifests/ray-head-deployment.yaml --ignore-not-found=true
    kubectl delete -f kubernetes/manifests/ray-storage.yaml --ignore-not-found=true
    kubectl delete -f monitoring/prometheus-deployment.yaml --ignore-not-found=true
    kubectl delete -f monitoring/grafana-deployment.yaml --ignore-not-found=true
    kubectl delete -f kubernetes/manifests/ray-namespace.yaml --ignore-not-found=true
    
    log_success "Cleanup completed"
}

# Main script
main() {
    case "${1:-deploy}" in
        "deploy")
            log_info "Starting Ray on Kubernetes deployment..."
            check_prerequisites
            create_namespace
            deploy_storage
            deploy_ray_cluster
            deploy_monitoring
            verify_deployment
            show_access_info
            log_success "Ray on Kubernetes deployment completed successfully!"
            ;;
        "cleanup")
            cleanup
            ;;
        "status")
            log_info "Checking Ray cluster status..."
            kubectl get pods -n $NAMESPACE
            kubectl get services -n $NAMESPACE
            ;;
        "logs")
            if [ -z "$2" ]; then
                log_error "Please specify a component (head, worker, prometheus, grafana)"
                exit 1
            fi
            kubectl logs -f deployment/ray-$2 -n $NAMESPACE
            ;;
        *)
            echo "Usage: $0 {deploy|cleanup|status|logs <component>}"
            echo ""
            echo "Commands:"
            echo "  deploy   - Deploy Ray cluster on Kubernetes"
            echo "  cleanup  - Remove Ray cluster from Kubernetes"
            echo "  status   - Show cluster status"
            echo "  logs     - Show logs for a component (head, worker, prometheus, grafana)"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
