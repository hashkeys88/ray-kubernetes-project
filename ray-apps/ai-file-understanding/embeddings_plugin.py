#!/usr/bin/env python3
"""
Ray-based Embeddings Plugin
Distributed embeddings generation for AI file understanding
"""

import ray
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import hashlib
import pickle
import os
from transformers import AutoTokenizer, AutoModel
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TextChunk:
    """Represents a chunk of text with its embedding"""
    text: str
    embedding: np.ndarray
    chunk_id: str
    file_id: str
    position: int
    metadata: Dict[str, Any] = None

@ray.remote
class EmbeddingsWorker:
    """Ray remote worker for computing embeddings"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        logger.info(f"EmbeddingsWorker initialized with model: {model_name} on {self.device}")
    
    def compute_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Compute embeddings for a batch of texts"""
        try:
            # Tokenize texts
            inputs = self.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Compute embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
                embeddings = embeddings.cpu().numpy()
            
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Error computing embeddings: {e}")
            raise

@ray.remote
class EmbeddingsCache:
    """Distributed cache for embeddings using Ray's object store"""
    
    def __init__(self):
        self.cache = {}
        logger.info("EmbeddingsCache initialized")
    
    def get_cache_key(self, file_id: str, text_hash: str) -> str:
        """Generate cache key for embeddings"""
        return f"{file_id}:{text_hash}"
    
    def get(self, cache_key: str) -> List[np.ndarray]:
        """Get embeddings from cache"""
        return self.cache.get(cache_key)
    
    def put(self, cache_key: str, embeddings: List[np.ndarray]):
        """Store embeddings in cache"""
        self.cache[cache_key] = embeddings
        logger.info(f"Cached embeddings for key: {cache_key}")
    
    def exists(self, cache_key: str) -> bool:
        """Check if embeddings exist in cache"""
        return cache_key in self.cache

class EmbeddingsPlugin:
    """Main embeddings plugin for processing files and generating embeddings"""
    
    def __init__(self, num_workers: int = 4, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.num_workers = num_workers
        self.model_name = model_name
        
        # Initialize Ray workers
        self.workers = [EmbeddingsWorker.remote(model_name) for _ in range(num_workers)]
        self.cache = EmbeddingsCache.remote()
        
        logger.info(f"EmbeddingsPlugin initialized with {num_workers} workers")
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 200 characters
                sentence_endings = ['.', '!', '?', '\n\n']
                for i in range(min(200, chunk_size)):
                    if text[end - i] in sentence_endings:
                        end = end - i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks
    
    def get_text_hash(self, text: str) -> str:
        """Generate hash for text content"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def process_file(self, file_id: str, text: str, metadata: Dict[str, Any] = None) -> List[TextChunk]:
        """Process a file and generate embeddings for its text chunks"""
        logger.info(f"Processing file: {file_id}")
        
        # Chunk the text
        text_chunks = self.chunk_text(text)
        logger.info(f"Split file into {len(text_chunks)} chunks")
        
        # Check cache for existing embeddings
        text_hash = self.get_text_hash(text)
        cache_key = ray.get(self.cache.get_cache_key.remote(file_id, text_hash))
        
        cached_embeddings = ray.get(self.cache.get.remote(cache_key))
        if cached_embeddings:
            logger.info(f"Using cached embeddings for file: {file_id}")
            embeddings = cached_embeddings
        else:
            # Compute embeddings in parallel
            logger.info(f"Computing embeddings for {len(text_chunks)} chunks")
            embeddings = self._compute_embeddings_parallel(text_chunks)
            
            # Cache the embeddings
            ray.get(self.cache.put.remote(cache_key, embeddings))
        
        # Create TextChunk objects
        chunks = []
        for i, (chunk_text, embedding) in enumerate(zip(text_chunks, embeddings)):
            chunk = TextChunk(
                text=chunk_text,
                embedding=np.array(embedding),
                chunk_id=f"{file_id}_chunk_{i}",
                file_id=file_id,
                position=i,
                metadata=metadata or {}
            )
            chunks.append(chunk)
        
        logger.info(f"Generated {len(chunks)} text chunks with embeddings for file: {file_id}")
        return chunks
    
    def _compute_embeddings_parallel(self, text_chunks: List[str]) -> List[np.ndarray]:
        """Compute embeddings using Ray workers in parallel"""
        batch_size = max(1, len(text_chunks) // self.num_workers)
        
        # Split chunks into batches
        batches = []
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i + batch_size]
            batches.append(batch)
        
        # Distribute batches to workers
        futures = []
        for i, batch in enumerate(batches):
            worker = self.workers[i % len(self.workers)]
            future = worker.compute_embeddings.remote(batch)
            futures.append(future)
        
        # Collect results
        results = ray.get(futures)
        
        # Flatten results
        embeddings = []
        for batch_embeddings in results:
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def compute_query_embedding(self, query: str) -> np.ndarray:
        """Compute embedding for a search query"""
        embeddings = self._compute_embeddings_parallel([query])
        return np.array(embeddings[0])
    
    def find_similar_chunks(self, query_embedding: np.ndarray, chunks: List[TextChunk], 
                          top_k: int = 50) -> List[Tuple[TextChunk, float]]:
        """Find most similar chunks to a query embedding"""
        if not chunks:
            return []
        
        # Compute similarities
        similarities = []
        for chunk in chunks:
            similarity = np.dot(query_embedding, chunk.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk.embedding)
            )
            similarities.append((chunk, float(similarity)))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

def process_multiple_files(files: List[Dict[str, Any]], 
                          embeddings_plugin: EmbeddingsPlugin) -> Dict[str, List[TextChunk]]:
    """Process multiple files and return their embeddings"""
    logger.info(f"Processing {len(files)} files")
    
    results = {}
    
    # Process files in parallel
    futures = []
    for file_data in files:
        file_id = file_data['id']
        text = file_data['text']
        metadata = file_data.get('metadata', {})
        
        future = ray.put(embeddings_plugin.process_file(file_id, text, metadata))
        futures.append((file_id, future))
    
    # Collect results
    for file_id, future in futures:
        chunks = ray.get(future)
        results[file_id] = chunks
    
    logger.info(f"Processed {len(results)} files successfully")
    return results

if __name__ == "__main__":
    # Initialize Ray
    ray.init(address='ray://ray-head-service:10001', ignore_reinit_error=True)
    
    # Test the embeddings plugin
    embeddings_plugin = EmbeddingsPlugin()
    
    # Sample text
    sample_text = """
    This is a sample document about machine learning and artificial intelligence.
    It contains multiple paragraphs with information about neural networks, deep learning,
    and various applications of AI in modern technology.
    
    Machine learning is a subset of artificial intelligence that focuses on algorithms
    that can learn from data without being explicitly programmed. Deep learning,
    on the other hand, uses neural networks with multiple layers to process data.
    
    Applications of AI include natural language processing, computer vision,
    robotics, and autonomous vehicles. These technologies are transforming
    various industries and creating new opportunities for innovation.
    """
    
    # Process the sample file
    chunks = embeddings_plugin.process_file("sample_doc_1", sample_text)
    
    print(f"Generated {len(chunks)} chunks with embeddings")
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        print(f"Chunk {i}: {chunk.text[:100]}...")
        print(f"Embedding shape: {chunk.embedding.shape}")
        print()
    
    ray.shutdown()
