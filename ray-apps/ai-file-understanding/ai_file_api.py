#!/usr/bin/env python3
"""
Ray Serve API for AI File Understanding
Complete API service for distributed file processing and understanding
"""

import ray
from ray import serve
import logging
from typing import Dict, List, Any, Optional, Union
import json
import time
import uuid
from dataclasses import asdict
import asyncio
from pydantic import BaseModel, Field

from multi_file_handler import MultiFileHandler, MultiFileRequest, MultiFileResponse
from embeddings_plugin import EmbeddingsPlugin
from summarization_plugin import SummarizationPlugin
from qa_plugin import QAPlugin
from file_processor import FileProcessingPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class SummarizeRequest(BaseModel):
    """Request model for summarization"""
    file_paths: List[str] = Field(..., description="List of file paths to summarize")
    max_summary_length: int = Field(800, description="Maximum length of summary")
    include_topics: bool = Field(True, description="Include extracted topics")
    include_followup_questions: bool = Field(True, description="Include follow-up questions")

class QARequest(BaseModel):
    """Request model for Q&A"""
    file_paths: List[str] = Field(..., description="List of file paths to search")
    query: str = Field(..., description="Question to answer")
    min_chunks: int = Field(5, description="Minimum number of chunks to use")
    max_chunks: int = Field(50, description="Maximum number of chunks to use")
    include_sources: bool = Field(True, description="Include source chunks")
    include_followup_questions: bool = Field(True, description="Include follow-up questions")

class CombinedRequest(BaseModel):
    """Request model for combined summarization and Q&A"""
    file_paths: List[str] = Field(..., description="List of file paths to process")
    query: str = Field(..., description="Question to answer")
    max_summary_length: int = Field(800, description="Maximum length of summary")
    min_chunks: int = Field(5, description="Minimum number of chunks for Q&A")
    max_chunks: int = Field(50, description="Maximum number of chunks for Q&A")

class FileInfoRequest(BaseModel):
    """Request model for file information"""
    file_path: str = Field(..., description="File path to analyze")

class BatchRequest(BaseModel):
    """Request model for batch processing"""
    requests: List[Dict[str, Any]] = Field(..., description="List of requests to process")
    operation: str = Field(..., description="Operation type: summarize, qa, or combined")

class APIResponse(BaseModel):
    """Standard API response model"""
    success: bool = Field(..., description="Whether the request was successful")
    data: Dict[str, Any] = Field(..., description="Response data")
    error: Optional[str] = Field(None, description="Error message if any")
    request_id: str = Field(..., description="Unique request identifier")
    processing_time: float = Field(..., description="Processing time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

@serve.deployment(
    name="ai_file_understanding",
    num_replicas=2,
    max_concurrent_queries=10
)
class AIFileUnderstandingAPI:
    """Main Ray Serve deployment for AI file understanding"""
    
    def __init__(self):
        self.multi_file_handler = MultiFileHandler(num_workers=4)
        self.embeddings_plugin = EmbeddingsPlugin()
        self.summarization_plugin = SummarizationPlugin()
        self.qa_plugin = QAPlugin(self.embeddings_plugin)
        self.file_pipeline = FileProcessingPipeline()
        
        logger.info("AIFileUnderstandingAPI initialized")
    
    async def __call__(self, request) -> Dict[str, Any]:
        """Main request handler"""
        try:
            # Parse request
            if hasattr(request, 'json'):
                request_data = await request.json()
            else:
                request_data = request
            
            endpoint = request_data.get('endpoint')
            data = request_data.get('data', {})
            
            # Route to appropriate handler
            if endpoint == 'summarize':
                return await self._handle_summarize(data)
            elif endpoint == 'qa':
                return await self._handle_qa(data)
            elif endpoint == 'combined':
                return await self._handle_combined(data)
            elif endpoint == 'file_info':
                return await self._handle_file_info(data)
            elif endpoint == 'batch':
                return await self._handle_batch(data)
            elif endpoint == 'health':
                return await self._handle_health()
            else:
                return self._error_response(f"Unknown endpoint: {endpoint}")
        
        except Exception as e:
            logger.error(f"Error in API handler: {e}")
            return self._error_response(str(e))
    
    async def _handle_summarize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle summarization requests"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Parse request
            request = SummarizeRequest(**data)
            
            # Create multi-file request
            multi_file_request = MultiFileRequest(
                file_paths=request.file_paths,
                operation='summarize',
                max_summary_length=request.max_summary_length
            )
            
            # Process request
            response = self.multi_file_handler.process_multi_file_request(multi_file_request)
            
            # Format response
            result_data = {
                'summary': response.results.get('summary', {}),
                'file_count': response.file_count,
                'success_count': response.success_count,
                'failure_count': response.failure_count
            }
            
            # Filter optional fields
            if not request.include_topics:
                result_data['summary'].pop('topics', None)
            if not request.include_followup_questions:
                result_data['summary'].pop('followup_questions', None)
            
            return APIResponse(
                success=True,
                data=result_data,
                request_id=request_id,
                processing_time=time.time() - start_time,
                metadata=response.metadata
            ).dict()
            
        except Exception as e:
            logger.error(f"Error in summarize handler: {e}")
            return self._error_response(str(e), request_id, time.time() - start_time)
    
    async def _handle_qa(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Q&A requests"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Parse request
            request = QARequest(**data)
            
            # Create multi-file request
            multi_file_request = MultiFileRequest(
                file_paths=request.file_paths,
                operation='qa',
                query=request.query,
                min_chunks=request.min_chunks,
                max_chunks=request.max_chunks
            )
            
            # Process request
            response = self.multi_file_handler.process_multi_file_request(multi_file_request)
            
            # Format response
            qa_result = response.results.get('qa', {})
            result_data = {
                'answer': qa_result.get('answer', ''),
                'confidence_score': qa_result.get('confidence_score', 0.0),
                'query_type': qa_result.get('query_type', 'unknown'),
                'file_count': response.file_count,
                'success_count': response.success_count,
                'failure_count': response.failure_count
            }
            
            # Add optional fields
            if request.include_sources:
                result_data['source_files'] = qa_result.get('source_files', [])
            if request.include_followup_questions:
                result_data['followup_questions'] = qa_result.get('followup_questions', [])
            
            return APIResponse(
                success=True,
                data=result_data,
                request_id=request_id,
                processing_time=time.time() - start_time,
                metadata=response.metadata
            ).dict()
            
        except Exception as e:
            logger.error(f"Error in Q&A handler: {e}")
            return self._error_response(str(e), request_id, time.time() - start_time)
    
    async def _handle_combined(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle combined summarization and Q&A requests"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Parse request
            request = CombinedRequest(**data)
            
            # Create multi-file request
            multi_file_request = MultiFileRequest(
                file_paths=request.file_paths,
                operation='both',
                query=request.query,
                max_summary_length=request.max_summary_length,
                min_chunks=request.min_chunks,
                max_chunks=request.max_chunks
            )
            
            # Process request
            response = self.multi_file_handler.process_multi_file_request(multi_file_request)
            
            # Format response
            result_data = {
                'summary': response.results.get('summary', {}),
                'qa': response.results.get('qa', {}),
                'file_count': response.file_count,
                'success_count': response.success_count,
                'failure_count': response.failure_count
            }
            
            return APIResponse(
                success=True,
                data=result_data,
                request_id=request_id,
                processing_time=time.time() - start_time,
                metadata=response.metadata
            ).dict()
            
        except Exception as e:
            logger.error(f"Error in combined handler: {e}")
            return self._error_response(str(e), request_id, time.time() - start_time)
    
    async def _handle_file_info(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle file information requests"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Parse request
            request = FileInfoRequest(**data)
            
            # Get file metadata
            metadata = self.file_pipeline.create_file_metadata(request.file_path)
            
            # Get supported formats
            supported_formats = self.file_pipeline.get_supported_formats()
            
            result_data = {
                'file_info': asdict(metadata),
                'supported_formats': supported_formats,
                'can_process': any(
                    processor.can_process(request.file_path, metadata.mime_type)
                    for processor in self.file_pipeline.workers[0].__class__().processors
                )
            }
            
            return APIResponse(
                success=True,
                data=result_data,
                request_id=request_id,
                processing_time=time.time() - start_time
            ).dict()
            
        except Exception as e:
            logger.error(f"Error in file info handler: {e}")
            return self._error_response(str(e), request_id, time.time() - start_time)
    
    async def _handle_batch(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle batch processing requests"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Parse request
            request = BatchRequest(**data)
            
            # Create multi-file requests
            multi_file_requests = []
            for req_data in request.requests:
                if request.operation == 'summarize':
                    req = SummarizeRequest(**req_data)
                    multi_file_request = MultiFileRequest(
                        file_paths=req.file_paths,
                        operation='summarize',
                        max_summary_length=req.max_summary_length
                    )
                elif request.operation == 'qa':
                    req = QARequest(**req_data)
                    multi_file_request = MultiFileRequest(
                        file_paths=req.file_paths,
                        operation='qa',
                        query=req.query,
                        min_chunks=req.min_chunks,
                        max_chunks=req.max_chunks
                    )
                else:
                    req = CombinedRequest(**req_data)
                    multi_file_request = MultiFileRequest(
                        file_paths=req.file_paths,
                        operation='both',
                        query=req.query,
                        max_summary_length=req.max_summary_length,
                        min_chunks=req.min_chunks,
                        max_chunks=req.max_chunks
                    )
                
                multi_file_requests.append(multi_file_request)
            
            # Process batch requests
            responses = self.multi_file_handler.batch_process_requests(multi_file_requests)
            
            # Format responses
            result_data = {
                'batch_results': [asdict(response) for response in responses],
                'total_requests': len(responses),
                'successful_requests': sum(1 for r in responses if r.success_count > 0),
                'failed_requests': sum(1 for r in responses if r.success_count == 0)
            }
            
            return APIResponse(
                success=True,
                data=result_data,
                request_id=request_id,
                processing_time=time.time() - start_time
            ).dict()
            
        except Exception as e:
            logger.error(f"Error in batch handler: {e}")
            return self._error_response(str(e), request_id, time.time() - start_time)
    
    async def _handle_health(self) -> Dict[str, Any]:
        """Handle health check requests"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Get system status
            stats = self.multi_file_handler.get_processing_stats()
            
            result_data = {
                'status': 'healthy',
                'system_stats': stats,
                'timestamp': time.time(),
                'version': '1.0.0'
            }
            
            return APIResponse(
                success=True,
                data=result_data,
                request_id=request_id,
                processing_time=time.time() - start_time
            ).dict()
            
        except Exception as e:
            logger.error(f"Error in health handler: {e}")
            return self._error_response(str(e), request_id, time.time() - start_time)
    
    def _error_response(self, error_message: str, request_id: str = None, 
                       processing_time: float = 0.0) -> Dict[str, Any]:
        """Create error response"""
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        return APIResponse(
            success=False,
            data={},
            error=error_message,
            request_id=request_id,
            processing_time=processing_time
        ).dict()

def setup_ray_serve():
    """Setup Ray Serve with the AI File Understanding API"""
    logger.info("Setting up Ray Serve for AI File Understanding...")
    
    # Initialize Ray
    ray.init(address='ray://ray-head-service:10001', ignore_reinit_error=True)
    
    # Start Ray Serve
    serve.start(detached=True, http_options={"host": "0.0.0.0", "port": 8000})
    
    # Deploy the API
    AIFileUnderstandingAPI.deploy(name="ai_file_understanding")
    
    logger.info("Ray Serve setup completed")
    logger.info("AI File Understanding API is available at: http://ray-head-service:8000")
    
    return serve.get_deployment("ai_file_understanding")

async def test_api():
    """Test the API with sample requests"""
    logger.info("Testing AI File Understanding API...")
    
    # Test health endpoint
    health_request = {"endpoint": "health", "data": {}}
    health_response = await AIFileUnderstandingAPI.get_handle().remote(health_request)
    print(f"Health check: {health_response}")
    
    # Test summarization endpoint
    summarize_request = {
        "endpoint": "summarize",
        "data": {
            "file_paths": ["/path/to/document1.pdf", "/path/to/document2.docx"],
            "max_summary_length": 500,
            "include_topics": True,
            "include_followup_questions": True
        }
    }
    # summarize_response = await AIFileUnderstandingAPI.get_handle().remote(summarize_request)
    # print(f"Summarization: {summarize_response}")
    
    logger.info("API testing completed")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='AI File Understanding API')
    parser.add_argument('--mode', choices=['setup', 'test'], default='setup',
                       help='Setup Ray Serve or run tests')
    
    args = parser.parse_args()
    
    if args.mode == 'setup':
        setup_ray_serve()
        
        # Keep the service running
        logger.info("AI File Understanding API is running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down Ray Serve...")
            serve.shutdown()
            ray.shutdown()
    else:
        asyncio.run(test_api())
