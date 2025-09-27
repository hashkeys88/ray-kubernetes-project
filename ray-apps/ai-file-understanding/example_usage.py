#!/usr/bin/env python3
"""
Example Usage of AI File Understanding System
Demonstrates how to use the Ray-based distributed AI file understanding system
"""

import ray
import asyncio
import logging
import tempfile
import os
from typing import List, Dict, Any

from multi_file_handler import MultiFileHandler, create_summarization_request, create_qa_request, create_combined_request
from embeddings_plugin import EmbeddingsPlugin
from summarization_plugin import SummarizationPlugin
from qa_plugin import QAPlugin
from file_processor import FileProcessingPipeline
from ai_file_api import AIFileUnderstandingAPI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_files() -> List[str]:
    """Create sample files for testing"""
    sample_files = []
    
    # Create sample text file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("""
        Machine Learning and Artificial Intelligence
        
        Machine learning is a subset of artificial intelligence that focuses on algorithms 
        that can learn from data without being explicitly programmed. It has revolutionized 
        many industries including healthcare, finance, and technology.
        
        Deep learning, a subset of machine learning, uses neural networks with multiple 
        layers to process complex data patterns. These networks can automatically learn 
        hierarchical representations of data, making them particularly effective for 
        tasks like image recognition and natural language processing.
        
        Applications of AI include autonomous vehicles, medical diagnosis, fraud detection,
        recommendation systems, and natural language understanding. The field continues
        to evolve rapidly with new architectures and techniques being developed regularly.
        """)
        sample_files.append(f.name)
    
    # Create another sample text file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("""
        Distributed Computing with Ray
        
        Ray is a distributed computing framework that makes it easy to scale Python 
        applications from a laptop to a cluster. It provides a simple API for 
        distributed computing and machine learning workloads.
        
        Key features of Ray include:
        - Ray Core: Distributed computing primitives
        - Ray Tune: Hyperparameter tuning
        - Ray Serve: Model serving
        - Ray RLlib: Reinforcement learning
        - Ray Train: Distributed training
        
        Ray is particularly useful for machine learning workloads that require
        distributed training, hyperparameter optimization, and model serving at scale.
        It integrates well with popular ML frameworks like PyTorch, TensorFlow, and
        scikit-learn.
        
        The framework provides fault tolerance, automatic scaling, and resource
        management, making it suitable for production environments.
        """)
        sample_files.append(f.name)
    
    # Create a third sample file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("""
        Kubernetes and Container Orchestration
        
        Kubernetes is an open-source container orchestration platform that automates
        the deployment, scaling, and management of containerized applications.
        
        Key concepts in Kubernetes include:
        - Pods: The smallest deployable units
        - Services: Network access to pods
        - Deployments: Manage pod replicas
        - ConfigMaps and Secrets: Configuration management
        - Namespaces: Resource isolation
        
        Kubernetes provides features like:
        - Automatic scaling based on demand
        - Rolling updates and rollbacks
        - Service discovery and load balancing
        - Resource quotas and limits
        - Health checks and self-healing
        
        The platform is widely used for deploying microservices, machine learning
        workloads, and cloud-native applications. It integrates with cloud providers
        and provides a consistent interface for managing applications across
        different environments.
        """)
        sample_files.append(f.name)
    
    return sample_files

async def example_basic_usage():
    """Basic usage example"""
    logger.info("=== Basic Usage Example ===")
    
    # Initialize Ray
    ray.init(address='ray://ray-head-service:10001', ignore_reinit_error=True)
    
    # Create sample files
    sample_files = create_sample_files()
    logger.info(f"Created {len(sample_files)} sample files")
    
    try:
        # Initialize components
        embeddings_plugin = EmbeddingsPlugin()
        summarization_plugin = SummarizationPlugin()
        qa_plugin = QAPlugin(embeddings_plugin)
        file_pipeline = FileProcessingPipeline()
        
        # Process files and generate embeddings
        logger.info("Processing files and generating embeddings...")
        file_chunks = {}
        
        for file_path in sample_files:
            result = file_pipeline.process_single_file(file_path)
            if result.success:
                chunks = embeddings_plugin.process_file(
                    result.file_id, 
                    result.text_content, 
                    result.metadata.__dict__
                )
                file_chunks[result.file_id] = chunks
                logger.info(f"Generated {len(chunks)} chunks for {result.file_id}")
        
        # Example 1: Summarize single file
        if file_chunks:
            first_file_id = list(file_chunks.keys())[0]
            summary_result = summarization_plugin.summarize_single_file(file_chunks[first_file_id])
            
            logger.info("=== Single File Summary ===")
            logger.info(f"Summary: {summary_result.summary}")
            logger.info(f"Topics: {summary_result.topics}")
            logger.info(f"Confidence: {summary_result.confidence_score:.2f}")
        
        # Example 2: Answer questions
        if file_chunks:
            all_chunks = []
            for chunks in file_chunks.values():
                all_chunks.extend(chunks)
            
            questions = [
                "What is machine learning?",
                "What are the key features of Ray?",
                "What is Kubernetes used for?"
            ]
            
            logger.info("=== Question Answering ===")
            for question in questions:
                qa_result = qa_plugin.answer_question(question, all_chunks)
                logger.info(f"Q: {question}")
                logger.info(f"A: {qa_result.answer}")
                logger.info(f"Confidence: {qa_result.confidence_score:.2f}")
                logger.info(f"Query Type: {qa_result.query_type}")
                logger.info("---")
    
    finally:
        # Clean up sample files
        for file_path in sample_files:
            if os.path.exists(file_path):
                os.unlink(file_path)
        
        ray.shutdown()

async def example_multi_file_handler():
    """Multi-file handler example"""
    logger.info("=== Multi-File Handler Example ===")
    
    # Initialize Ray
    ray.init(address='ray://ray-head-service:10001', ignore_reinit_error=True)
    
    # Create sample files
    sample_files = create_sample_files()
    
    try:
        # Initialize multi-file handler
        handler = MultiFileHandler(num_workers=2)
        
        # Example 1: Summarize multiple files
        logger.info("=== Multi-File Summarization ===")
        summarize_request = create_summarization_request(sample_files, max_summary_length=600)
        summarize_response = handler.process_multi_file_request(summarize_request)
        
        logger.info(f"Summarization completed in {summarize_response.processing_time:.2f}s")
        logger.info(f"Processed {summarize_response.success_count}/{summarize_response.file_count} files")
        
        if 'summary' in summarize_response.results:
            summary_data = summarize_response.results['summary']
            logger.info(f"Summary: {summary_data['summary']}")
            logger.info(f"Topics: {summary_data['topics']}")
            logger.info(f"Confidence: {summary_data['confidence_score']:.2f}")
        
        # Example 2: Q&A across multiple files
        logger.info("=== Multi-File Q&A ===")
        qa_request = create_qa_request(
            sample_files, 
            "What are the main topics discussed across all documents?",
            min_chunks=5,
            max_chunks=30
        )
        qa_response = handler.process_multi_file_request(qa_request)
        
        logger.info(f"Q&A completed in {qa_response.processing_time:.2f}s")
        
        if 'qa' in qa_response.results:
            qa_data = qa_response.results['qa']
            logger.info(f"Answer: {qa_data['answer']}")
            logger.info(f"Confidence: {qa_data['confidence_score']:.2f}")
            logger.info(f"Query Type: {qa_data['query_type']}")
            logger.info(f"Source Files: {qa_data['source_files']}")
        
        # Example 3: Combined operation
        logger.info("=== Combined Summarization and Q&A ===")
        combined_request = create_combined_request(
            sample_files,
            "How do these technologies relate to each other?",
            max_summary_length=500
        )
        combined_response = handler.process_multi_file_request(combined_request)
        
        logger.info(f"Combined operation completed in {combined_response.processing_time:.2f}s")
        
        if 'summary' in combined_response.results and 'qa' in combined_response.results:
            logger.info("=== Summary ===")
            logger.info(combined_response.results['summary']['summary'])
            
            logger.info("=== Q&A ===")
            logger.info(combined_response.results['qa']['answer'])
    
    finally:
        # Clean up sample files
        for file_path in sample_files:
            if os.path.exists(file_path):
                os.unlink(file_path)
        
        ray.shutdown()

async def example_api_usage():
    """API usage example"""
    logger.info("=== API Usage Example ===")
    
    # Initialize Ray
    ray.init(address='ray://ray-head-service:10001', ignore_reinit_error=True)
    
    # Create sample files
    sample_files = create_sample_files()
    
    try:
        # Setup Ray Serve (in a real deployment, this would be done separately)
        from ray import serve
        serve.start(detached=True, http_options={"host": "0.0.0.0", "port": 8000})
        
        # Deploy the API
        AIFileUnderstandingAPI.deploy(name="ai_file_understanding")
        
        # Get the API handle
        api_handle = serve.get_deployment("ai_file_understanding").get_handle()
        
        # Example API calls
        logger.info("=== API Health Check ===")
        health_request = {"endpoint": "health", "data": {}}
        health_response = await api_handle.remote(health_request)
        logger.info(f"Health Response: {health_response}")
        
        # Example: Summarization via API
        logger.info("=== API Summarization ===")
        summarize_request = {
            "endpoint": "summarize",
            "data": {
                "file_paths": sample_files,
                "max_summary_length": 400,
                "include_topics": True,
                "include_followup_questions": True
            }
        }
        # Note: In a real scenario, you'd make HTTP requests to the API
        # For this example, we'll simulate the API call
        logger.info("API summarization request prepared")
        
        # Example: Q&A via API
        logger.info("=== API Q&A ===")
        qa_request = {
            "endpoint": "qa",
            "data": {
                "file_paths": sample_files,
                "query": "What are the key technologies mentioned?",
                "min_chunks": 3,
                "max_chunks": 20,
                "include_sources": True,
                "include_followup_questions": True
            }
        }
        logger.info("API Q&A request prepared")
        
        # Shutdown serve
        serve.shutdown()
    
    finally:
        # Clean up sample files
        for file_path in sample_files:
            if os.path.exists(file_path):
                os.unlink(file_path)
        
        ray.shutdown()

def example_power_law_dynamics():
    """Demonstrate power law dynamics in action"""
    logger.info("=== Power Law Dynamics Example ===")
    
    # Initialize Ray
    ray.init(address='ray://ray-head-service:10001', ignore_reinit_error=True)
    
    try:
        from multi_file_handler import MultiFileProcessor
        
        # Create a processor to demonstrate power law dynamics
        processor = MultiFileProcessor.remote()
        
        # Simulate different query types and their score distributions
        logger.info("Demonstrating power law dynamics for different query types...")
        
        # Direct question (steep curve)
        direct_similarities = [
            ("chunk1", 0.95), ("chunk2", 0.92), ("chunk3", 0.88),
            ("chunk4", 0.45), ("chunk5", 0.42), ("chunk6", 0.38),
            ("chunk7", 0.35), ("chunk8", 0.32), ("chunk9", 0.28)
        ]
        
        # Broad question (flatter curve)
        broad_similarities = [
            ("chunk1", 0.75), ("chunk2", 0.72), ("chunk3", 0.68),
            ("chunk4", 0.65), ("chunk5", 0.62), ("chunk6", 0.58),
            ("chunk7", 0.55), ("chunk8", 0.52), ("chunk9", 0.48)
        ]
        
        # Apply power law filtering
        direct_filtered = ray.get(processor.apply_power_law_dynamics.remote(direct_similarities, 'direct'))
        broad_filtered = ray.get(processor.apply_power_law_dynamics.remote(broad_similarities, 'broad'))
        
        logger.info(f"Direct question: {len(direct_similarities)} -> {len(direct_filtered)} chunks")
        logger.info(f"Broad question: {len(broad_similarities)} -> {len(broad_filtered)} chunks")
        
        logger.info("Power law dynamics successfully demonstrated!")
    
    finally:
        ray.shutdown()

async def main():
    """Run all examples"""
    logger.info("Starting AI File Understanding Examples")
    
    try:
        await example_basic_usage()
        await example_multi_file_handler()
        await example_api_usage()
        example_power_law_dynamics()
        
        logger.info("All examples completed successfully!")
    
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
