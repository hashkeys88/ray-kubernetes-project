#!/bin/bash

# Ray Examples Runner Script
# Run sample Ray applications on the deployed cluster

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="ray-system"
RAY_ADDRESS="ray://ray-head-service:10001"

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

check_ray_cluster() {
    log_info "Checking Ray cluster status..."
    
    if ! kubectl get pods -l app=ray-head -n $NAMESPACE | grep -q "Running"; then
        log_error "Ray head node is not running. Please deploy the cluster first."
        exit 1
    fi
    
    if ! kubectl get pods -l app=ray-worker -n $NAMESPACE | grep -q "Running"; then
        log_warning "Ray worker nodes are not running. Some examples may not work properly."
    fi
    
    log_success "Ray cluster is accessible"
}

run_distributed_training() {
    log_info "Running distributed training example..."
    
    # Create a temporary pod to run the training script
    kubectl run ray-training-$(date +%s) \
        --image=rayproject/ray:2.8.0-py310 \
        --rm -i --restart=Never \
        --namespace=$NAMESPACE \
        --command -- python3 -c "
import sys
sys.path.append('/tmp')
exec(open('/tmp/distributed_training.py').read())
" || log_warning "Distributed training example completed with warnings"
    
    log_success "Distributed training example completed"
}

run_hyperparameter_tuning() {
    log_info "Running hyperparameter tuning example..."
    
    # Create a temporary pod to run the tuning script
    kubectl run ray-tuning-$(date +%s) \
        --image=rayproject/ray:2.8.0-py310 \
        --rm -i --restart=Never \
        --namespace=$NAMESPACE \
        --command -- python3 -c "
import sys
sys.path.append('/tmp')
exec(open('/tmp/ray_tune_example.py').read())
" || log_warning "Hyperparameter tuning example completed with warnings"
    
    log_success "Hyperparameter tuning example completed"
}

run_ray_serve() {
    log_info "Running Ray Serve example..."
    
    # Create a temporary pod to run the serve script
    kubectl run ray-serve-$(date +%s) \
        --image=rayproject/ray:2.8.0-py310 \
        --rm -i --restart=Never \
        --namespace=$NAMESPACE \
        --command -- python3 -c "
import sys
sys.path.append('/tmp')
exec(open('/tmp/ray_serve_example.py').read())
" || log_warning "Ray Serve example completed with warnings"
    
    log_success "Ray Serve example completed"
}

run_ai_file_understanding() {
    log_info "Running AI File Understanding example..."
    
    # Create a temporary pod to run the AI file understanding script
    kubectl run ray-ai-files-$(date +%s) \
        --image=rayproject/ray:2.8.0-py310 \
        --rm -i --restart=Never \
        --namespace=$NAMESPACE \
        --command -- python3 -c "
import sys
sys.path.append('/tmp')
exec(open('/tmp/ai_file_api.py').read())
" || log_warning "AI File Understanding example completed with warnings"
    
    log_success "AI File Understanding example completed"
}

run_ai_examples() {
    log_info "Running AI File Understanding examples..."
    
    # Create a temporary pod to run the examples
    kubectl run ray-ai-examples-$(date +%s) \
        --image=rayproject/ray:2.8.0-py310 \
        --rm -i --restart=Never \
        --namespace=$NAMESPACE \
        --command -- python3 -c "
import sys
sys.path.append('/tmp')
exec(open('/tmp/example_usage.py').read())
" || log_warning "AI File Understanding examples completed with warnings"
    
    log_success "AI File Understanding examples completed"
}

run_custom_script() {
    local script_path="$1"
    
    if [ ! -f "$script_path" ]; then
        log_error "Script file not found: $script_path"
        exit 1
    fi
    
    log_info "Running custom script: $script_path"
    
    # Get the script name without path
    local script_name=$(basename "$script_path")
    
    # Create a temporary pod to run the custom script
    kubectl run ray-custom-$(date +%s) \
        --image=rayproject/ray:2.8.0-py310 \
        --rm -i --restart=Never \
        --namespace=$NAMESPACE \
        --command -- python3 -c "
import sys
sys.path.append('/tmp')
exec(open('/tmp/$script_name').read())
" || log_warning "Custom script completed with warnings"
    
    log_success "Custom script completed"
}

show_ray_dashboard() {
    log_info "Ray Dashboard access information:"
    echo ""
    echo "To access the Ray Dashboard:"
    echo "1. kubectl port-forward -n $NAMESPACE service/ray-head-service 8265:8265"
    echo "2. Open http://localhost:8265 in your browser"
    echo ""
    echo "Dashboard features:"
    echo "- Cluster overview and node status"
    echo "- Task and actor monitoring"
    echo "- Resource utilization"
    echo "- Job submission and management"
    echo ""
}

show_monitoring() {
    log_info "Monitoring access information:"
    echo ""
    echo "Prometheus (Metrics):"
    echo "1. kubectl port-forward -n $NAMESPACE service/prometheus-service 9090:9090"
    echo "2. Open http://localhost:9090 in your browser"
    echo ""
    echo "Grafana (Dashboards):"
    echo "1. kubectl port-forward -n $NAMESPACE service/grafana-service 3000:3000"
    echo "2. Open http://localhost:3000 in your browser"
    echo "3. Login with admin/admin123"
    echo ""
}

# Main script
main() {
    case "${1:-help}" in
        "training")
            check_ray_cluster
            run_distributed_training
            ;;
        "tuning")
            check_ray_cluster
            run_hyperparameter_tuning
            ;;
        "serve")
            check_ray_cluster
            run_ray_serve
            ;;
        "all")
            check_ray_cluster
            run_distributed_training
            run_hyperparameter_tuning
            run_ray_serve
            run_ai_file_understanding
            ;;
        "ai-files")
            check_ray_cluster
            run_ai_file_understanding
            ;;
        "ai-examples")
            check_ray_cluster
            run_ai_examples
            ;;
        "custom")
            if [ -z "$2" ]; then
                log_error "Please specify a script path"
                exit 1
            fi
            check_ray_cluster
            run_custom_script "$2"
            ;;
        "dashboard")
            show_ray_dashboard
            ;;
        "monitoring")
            show_monitoring
            ;;
        "help"|*)
            echo "Usage: $0 {training|tuning|serve|ai-files|ai-examples|all|custom <script>|dashboard|monitoring}"
            echo ""
            echo "Commands:"
            echo "  training     - Run distributed training example"
            echo "  tuning       - Run hyperparameter tuning example"
            echo "  serve        - Run Ray Serve example"
            echo "  ai-files     - Run AI File Understanding API example"
            echo "  ai-examples  - Run AI File Understanding examples"
            echo "  all          - Run all examples"
            echo "  custom       - Run a custom script"
            echo "  dashboard    - Show Ray Dashboard access info"
            echo "  monitoring   - Show monitoring access info"
            echo ""
            echo "Examples:"
            echo "  $0 training"
            echo "  $0 ai-files"
            echo "  $0 ai-examples"
            echo "  $0 custom /path/to/your/script.py"
            echo "  $0 dashboard"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
