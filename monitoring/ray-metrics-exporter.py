#!/usr/bin/env python3
"""
Ray Metrics Exporter for Prometheus
Custom metrics exporter for Ray cluster monitoring
"""

import ray
import time
import logging
from prometheus_client import start_http_server, Gauge, Counter, Histogram
from typing import Dict, Any
import psutil
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
ray_nodes_total = Gauge('ray_nodes_total', 'Total number of Ray nodes')
ray_tasks_total = Counter('ray_tasks_total', 'Total number of tasks executed')
ray_actors_total = Gauge('ray_actors_total', 'Total number of actors')
ray_objects_total = Gauge('ray_objects_total', 'Total number of objects in object store')
ray_memory_used_bytes = Gauge('ray_memory_used_bytes', 'Memory used by Ray in bytes')
ray_memory_total_bytes = Gauge('ray_memory_total_bytes', 'Total memory available to Ray in bytes')
ray_cpu_usage_percent = Gauge('ray_cpu_usage_percent', 'CPU usage percentage')
ray_gpu_usage_percent = Gauge('ray_gpu_usage_percent', 'GPU usage percentage')
ray_task_duration_seconds = Histogram('ray_task_duration_seconds', 'Task execution duration in seconds')

class RayMetricsExporter:
    """Ray metrics exporter for Prometheus"""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.ray_initialized = False
        self.start_time = time.time()
        
    def initialize_ray(self):
        """Initialize Ray connection"""
        try:
            ray.init(address='ray://ray-head-service:10001', ignore_reinit_error=True)
            self.ray_initialized = True
            logger.info("Ray initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Ray: {e}")
            self.ray_initialized = False
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get Ray cluster statistics"""
        if not self.ray_initialized:
            return {}
        
        try:
            # Get cluster resources
            cluster_resources = ray.cluster_resources()
            
            # Get node stats
            nodes = ray.nodes()
            
            # Get task stats (if available)
            task_stats = {}
            try:
                # This is a simplified version - in practice you'd need to track tasks
                task_stats = {"total_tasks": 0}
            except:
                pass
            
            return {
                "cluster_resources": cluster_resources,
                "nodes": nodes,
                "task_stats": task_stats
            }
        except Exception as e:
            logger.error(f"Failed to get cluster stats: {e}")
            return {}
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system-level statistics"""
        try:
            # Memory stats
            memory = psutil.virtual_memory()
            
            # CPU stats
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Process stats for Ray processes
            ray_memory = 0
            ray_cpu = 0
            
            for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cpu_percent']):
                try:
                    if 'ray' in proc.info['name'].lower():
                        ray_memory += proc.info['memory_info'].rss
                        ray_cpu += proc.info['cpu_percent']
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return {
                "memory_total": memory.total,
                "memory_used": memory.used,
                "memory_percent": memory.percent,
                "cpu_percent": cpu_percent,
                "ray_memory": ray_memory,
                "ray_cpu": ray_cpu
            }
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {}
    
    def update_metrics(self):
        """Update Prometheus metrics"""
        try:
            # Get cluster stats
            cluster_stats = self.get_cluster_stats()
            
            # Get system stats
            system_stats = self.get_system_stats()
            
            # Update node metrics
            if cluster_stats and "nodes" in cluster_stats:
                active_nodes = len([node for node in cluster_stats["nodes"] if node["Alive"]])
                ray_nodes_total.set(active_nodes)
            
            # Update resource metrics
            if cluster_stats and "cluster_resources" in cluster_stats:
                resources = cluster_stats["cluster_resources"]
                
                # Memory metrics
                if "memory" in resources:
                    ray_memory_total_bytes.set(resources["memory"])
                
                # CPU metrics
                if "CPU" in resources:
                    ray_cpu_usage_percent.set(system_stats.get("ray_cpu", 0))
            
            # Update system metrics
            if system_stats:
                ray_memory_used_bytes.set(system_stats.get("ray_memory", 0))
            
            # Update task metrics (simplified)
            ray_tasks_total.inc(0)  # In practice, you'd track actual task completions
            
            logger.debug("Metrics updated successfully")
            
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")
    
    def run(self):
        """Run the metrics exporter"""
        logger.info(f"Starting Ray metrics exporter on port {self.port}")
        
        # Initialize Ray
        self.initialize_ray()
        
        # Start Prometheus HTTP server
        start_http_server(self.port)
        logger.info(f"Prometheus metrics server started on port {self.port}")
        
        # Main loop
        while True:
            try:
                self.update_metrics()
                time.sleep(10)  # Update every 10 seconds
            except KeyboardInterrupt:
                logger.info("Shutting down metrics exporter...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(5)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ray Metrics Exporter')
    parser.add_argument('--port', type=int, default=8080, help='Port for metrics server')
    parser.add_argument('--ray-address', type=str, default='ray://ray-head-service:10001',
                       help='Ray cluster address')
    
    args = parser.parse_args()
    
    exporter = RayMetricsExporter(port=args.port)
    exporter.run()

if __name__ == "__main__":
    main()
