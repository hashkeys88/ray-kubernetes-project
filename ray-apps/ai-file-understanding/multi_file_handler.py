#!/usr/bin/env python3
"""
Multi-File Handler with Power Law Dynamics
Handles multiple files simultaneously with intelligent chunk selection
"""

import ray
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import time
from collections import defaultdict

from embeddings_plugin import TextChunk, EmbeddingsPlugin
from summarization_plugin import SummarizationPlugin, SummaryResult
from qa_plugin import QAPlugin, QAResult
from file_processor import FileProcessingPipeline, ProcessingResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MultiFileRequest:
    """Request for multi-file processing"""
    file_paths: List[str]
    operation: str  # 'summarize', 'qa', 'both'
    query: Optional[str] = None
    max_summary_length: int = 800
    min_chunks: int = 5
    max_chunks: int = 50
    metadata: Dict[str, Any] = None

@dataclass
class MultiFileResponse:
    """Response from multi-file processing"""
    request_id: str
    operation: str
    results: Dict[str, Any]
    processing_time: float
    file_count: int
    success_count: int
    failure_count: int
    metadata: Dict[str, Any] = None

@ray.remote
class MultiFileProcessor:
    """Ray remote worker for multi-file processing"""
    
    def __init__(self):
        self.embeddings_plugin = EmbeddingsPlugin()
        self.summarization_plugin = SummarizationPlugin()
        self.qa_plugin = QAPlugin(self.embeddings_plugin)
        self.file_pipeline = FileProcessingPipeline()
        
        logger.info("MultiFileProcessor initialized")
    
    def process_files_for_understanding(self, file_paths: List[str]) -> Dict[str, List[TextChunk]]:
        """Process multiple files and return their embeddings"""
        logger.info(f"Processing {len(file_paths)} files for understanding")
        
        # Step 1: Process files to extract text
        processing_results = self.file_pipeline.process_multiple_files(file_paths)
        
        # Step 2: Generate embeddings for successful files
        file_chunks = {}
        for file_id, result in processing_results.items():
            if result.success and result.text_content.strip():
                try:
                    chunks = self.embeddings_plugin.process_file(
                        file_id, 
                        result.text_content, 
                        result.metadata.__dict__
                    )
                    file_chunks[file_id] = chunks
                    logger.info(f"Generated {len(chunks)} chunks for file {file_id}")
                except Exception as e:
                    logger.error(f"Error generating embeddings for {file_id}: {e}")
        
        return file_chunks
    
    def apply_power_law_dynamics(self, similarities: List[Tuple[TextChunk, float]], 
                               query_type: str) -> List[Tuple[TextChunk, float]]:
        """Apply power law dynamics to filter chunks based on query type"""
        if not similarities:
            return []
        
        # Take top 50 chunks for analysis (as per Dropbox approach)
        top_chunks = similarities[:50]
        
        if len(top_chunks) <= 5:
            return top_chunks
        
        # Calculate relevance scores
        scores = [score for _, score in top_chunks]
        max_score = max(scores)
        min_score = min(scores)
        
        # Apply power law filtering (cut off bottom 20% of the spread)
        score_spread = max_score - min_score
        cutoff_threshold = min_score + (0.2 * score_spread)
        
        # Filter chunks above threshold
        filtered_chunks = [(chunk, score) for chunk, score in top_chunks 
                          if score >= cutoff_threshold]
        
        # Determine final chunk count based on query type
        if query_type == 'direct':
            # Direct questions: fewer, more relevant chunks
            final_chunks = filtered_chunks[:15]
        else:
            # Broad questions: more comprehensive coverage
            final_chunks = filtered_chunks[:40]
        
        logger.info(f"Power law filtering: {len(similarities)} -> {len(filtered_chunks)} -> {len(final_chunks)} chunks")
        return final_chunks
    
    def determine_query_type_from_context(self, similarities: List[Tuple[TextChunk, float]]) -> str:
        """Determine if query is direct or broad based on score distribution"""
        if len(similarities) < 3:
            return 'direct'
        
        scores = [score for _, score in similarities[:20]]  # Look at top 20
        
        # Calculate steepness of the power law curve
        top_20_percent = int(len(scores) * 0.2)
        bottom_20_percent = int(len(scores) * 0.2)
        
        if top_20_percent == 0 or bottom_20_percent == 0:
            return 'direct'
        
        top_scores = scores[:top_20_percent]
        bottom_scores = scores[-bottom_20_percent:]
        
        top_avg = np.mean(top_scores)
        bottom_avg = np.mean(bottom_scores)
        
        # If there's a steep drop-off, it's a direct question
        steepness = (top_avg - bottom_avg) / top_avg if top_avg > 0 else 0
        
        query_type = 'direct' if steepness > 0.3 else 'broad'
        logger.info(f"Query type determined: {query_type} (steepness: {steepness:.3f})")
        
        return query_type

class MultiFileHandler:
    """Main handler for multi-file operations with power law dynamics"""
    
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.workers = [MultiFileProcessor.remote() for _ in range(num_workers)]
        
        logger.info(f"MultiFileHandler initialized with {num_workers} workers")
    
    def process_multi_file_request(self, request: MultiFileRequest) -> MultiFileResponse:
        """Process a multi-file request with power law dynamics"""
        start_time = time.time()
        request_id = f"req_{int(start_time)}"
        
        logger.info(f"Processing multi-file request {request_id}: {request.operation} on {len(request.file_paths)} files")
        
        try:
            # Step 1: Process files and generate embeddings
            file_chunks = self._process_files_parallel(request.file_paths)
            
            if not file_chunks:
                return MultiFileResponse(
                    request_id=request_id,
                    operation=request.operation,
                    results={"error": "No files could be processed successfully"},
                    processing_time=time.time() - start_time,
                    file_count=len(request.file_paths),
                    success_count=0,
                    failure_count=len(request.file_paths)
                )
            
            # Step 2: Perform requested operations
            results = {}
            
            if request.operation in ['summarize', 'both']:
                summary_result = self._summarize_multiple_files(file_chunks, request.max_summary_length)
                results['summary'] = {
                    'summary': summary_result.summary,
                    'topics': summary_result.topics,
                    'confidence_score': summary_result.confidence_score,
                    'source_files': list(set(chunk.file_id for chunk in summary_result.chunk_sources)),
                    'followup_questions': summary_result.metadata.get('followup_questions', [])
                }
            
            if request.operation in ['qa', 'both'] and request.query:
                qa_result = self._answer_question_multiple_files(
                    request.query, file_chunks, request.min_chunks, request.max_chunks
                )
                results['qa'] = {
                    'answer': qa_result.answer,
                    'confidence_score': qa_result.confidence_score,
                    'query_type': qa_result.query_type,
                    'source_files': list(set(chunk.file_id for chunk in qa_result.source_chunks)),
                    'followup_questions': qa_result.followup_questions
                }
            
            processing_time = time.time() - start_time
            
            return MultiFileResponse(
                request_id=request_id,
                operation=request.operation,
                results=results,
                processing_time=processing_time,
                file_count=len(request.file_paths),
                success_count=len(file_chunks),
                failure_count=len(request.file_paths) - len(file_chunks),
                metadata={
                    'total_chunks': sum(len(chunks) for chunks in file_chunks.values()),
                    'processed_files': list(file_chunks.keys())
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing multi-file request {request_id}: {e}")
            return MultiFileResponse(
                request_id=request_id,
                operation=request.operation,
                results={"error": str(e)},
                processing_time=time.time() - start_time,
                file_count=len(request.file_paths),
                success_count=0,
                failure_count=len(request.file_paths)
            )
    
    def _process_files_parallel(self, file_paths: List[str]) -> Dict[str, List[TextChunk]]:
        """Process multiple files in parallel"""
        # Distribute files across workers
        batch_size = max(1, len(file_paths) // self.num_workers)
        batches = [file_paths[i:i + batch_size] for i in range(0, len(file_paths), batch_size)]
        
        futures = []
        for i, batch in enumerate(batches):
            worker = self.workers[i % len(self.workers)]
            future = worker.process_files_for_understanding.remote(batch)
            futures.append(future)
        
        # Collect results
        batch_results = ray.get(futures)
        
        # Combine results
        all_file_chunks = {}
        for batch_chunks in batch_results:
            all_file_chunks.update(batch_chunks)
        
        logger.info(f"Processed {len(all_file_chunks)} files successfully")
        return all_file_chunks
    
    def _summarize_multiple_files(self, file_chunks: Dict[str, List[TextChunk]], 
                                 max_summary_length: int) -> SummaryResult:
        """Summarize multiple files using combined approach"""
        # Use the first worker's summarization plugin
        worker = self.workers[0]
        
        # Combine all chunks for clustering
        all_chunks = []
        for file_id, chunks in file_chunks.items():
            for chunk in chunks:
                chunk.file_id = file_id
                all_chunks.append(chunk)
        
        # Use summarization plugin with increased clusters for multiple files
        summary_result = ray.get(worker.summarization_plugin.summarize_multiple_files.remote(
            file_chunks, max_summary_length
        ))
        
        return summary_result
    
    def _answer_question_multiple_files(self, query: str, file_chunks: Dict[str, List[TextChunk]],
                                       min_chunks: int, max_chunks: int) -> QAResult:
        """Answer question across multiple files with power law dynamics"""
        worker = self.workers[0]
        
        # Combine all chunks from all files
        all_chunks = []
        for file_id, chunks in file_chunks.items():
            for chunk in chunks:
                chunk.file_id = file_id
                all_chunks.append(chunk)
        
        # Compute query embedding
        query_embedding = ray.get(worker.embeddings_plugin.compute_query_embedding.remote(query))
        
        # Calculate similarities for all chunks
        similarities = ray.get(worker.embeddings_plugin.find_similar_chunks.remote(
            query_embedding, all_chunks, top_k=50
        ))
        
        # Apply power law dynamics
        filtered_chunks = ray.get(worker.apply_power_law_dynamics.remote(similarities, 'unknown'))
        
        # Determine query type from filtered results
        query_type = ray.get(worker.determine_query_type_from_context.remote(similarities))
        
        # Apply final filtering based on query type
        final_chunks = ray.get(worker.apply_power_law_dynamics.remote(similarities, query_type))
        
        relevant_chunks = [chunk for chunk, score in final_chunks]
        
        # Generate answer using QA plugin
        qa_result = ray.get(worker.qa_plugin.answer_question_multiple_files.remote(
            query, file_chunks, min_chunks, max_chunks
        ))
        
        # Update with power law filtered chunks
        qa_result.source_chunks = relevant_chunks
        qa_result.query_type = query_type
        
        return qa_result
    
    def batch_process_requests(self, requests: List[MultiFileRequest]) -> List[MultiFileResponse]:
        """Process multiple requests in parallel"""
        logger.info(f"Processing {len(requests)} requests in batch")
        
        # Distribute requests across workers
        futures = []
        for i, request in enumerate(requests):
            worker = self.workers[i % len(self.workers)]
            future = ray.put(self.process_multi_file_request(request))  # Use local method
            futures.append(future)
        
        # Collect results
        results = ray.get(futures)
        
        logger.info(f"Completed batch processing of {len(requests)} requests")
        return results
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "num_workers": self.num_workers,
            "worker_status": "active",
            "supported_operations": ["summarize", "qa", "both"],
            "power_law_enabled": True
        }

# Utility functions for easy integration
def create_summarization_request(file_paths: List[str], max_summary_length: int = 800) -> MultiFileRequest:
    """Create a summarization request for multiple files"""
    return MultiFileRequest(
        file_paths=file_paths,
        operation='summarize',
        max_summary_length=max_summary_length
    )

def create_qa_request(file_paths: List[str], query: str, 
                     min_chunks: int = 5, max_chunks: int = 50) -> MultiFileRequest:
    """Create a Q&A request for multiple files"""
    return MultiFileRequest(
        file_paths=file_paths,
        operation='qa',
        query=query,
        min_chunks=min_chunks,
        max_chunks=max_chunks
    )

def create_combined_request(file_paths: List[str], query: str, 
                           max_summary_length: int = 800) -> MultiFileRequest:
    """Create a combined summarization and Q&A request"""
    return MultiFileRequest(
        file_paths=file_paths,
        operation='both',
        query=query,
        max_summary_length=max_summary_length
    )

if __name__ == "__main__":
    # Initialize Ray
    ray.init(address='ray://ray-head-service:10001', ignore_reinit_error=True)
    
    # Test the multi-file handler
    handler = MultiFileHandler()
    
    # Create test files (in real usage, these would be actual file paths)
    test_files = [
        "/path/to/document1.pdf",
        "/path/to/document2.docx",
        "/path/to/video1.mp4",
        "/path/to/audio1.mp3"
    ]
    
    # Test summarization
    summarize_request = create_summarization_request(test_files)
    print("Testing multi-file summarization...")
    # result = handler.process_multi_file_request(summarize_request)
    # print(f"Summary: {result.results.get('summary', {}).get('summary', 'N/A')}")
    
    # Test Q&A
    qa_request = create_qa_request(test_files, "What are the main topics discussed?")
    print("Testing multi-file Q&A...")
    # qa_result = handler.process_multi_file_request(qa_request)
    # print(f"Answer: {qa_result.results.get('qa', {}).get('answer', 'N/A')}")
    
    print("Multi-file handler test completed")
    print("Processing stats:", handler.get_processing_stats())
    
    ray.shutdown()
