#!/usr/bin/env python3
"""
Ray-based Summarization Plugin
Implements k-means clustering approach for intelligent summarization
"""

import ray
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from sklearn.cluster import KMeans
from dataclasses import dataclass
import openai
import os
from embeddings_plugin import TextChunk, EmbeddingsPlugin

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SummaryResult:
    """Result of summarization process"""
    summary: str
    topics: List[str]
    chunk_sources: List[TextChunk]
    confidence_score: float
    metadata: Dict[str, Any] = None

@ray.remote
class ClusteringWorker:
    """Ray remote worker for k-means clustering"""
    
    def __init__(self):
        self.kmeans = None
        logger.info("ClusteringWorker initialized")
    
    def cluster_chunks(self, embeddings: np.ndarray, n_clusters: int = None) -> np.ndarray:
        """Perform k-means clustering on embeddings"""
        try:
            # Determine optimal number of clusters if not specified
            if n_clusters is None:
                n_clusters = min(max(3, len(embeddings) // 10), 20)  # Between 3 and 20 clusters
            
            # Perform k-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            logger.info(f"Clustered {len(embeddings)} embeddings into {n_clusters} clusters")
            return cluster_labels
            
        except Exception as e:
            logger.error(f"Error in clustering: {e}")
            raise

@ray.remote
class LLMWorker:
    """Ray remote worker for LLM operations"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        
        if self.api_key:
            openai.api_key = self.api_key
        
        logger.info(f"LLMWorker initialized with model: {model}")
    
    def generate_summary(self, context: str, max_length: int = 500) -> str:
        """Generate summary using LLM"""
        try:
            if not self.api_key:
                # Fallback to simple extractive summary
                return self._extractive_summary(context, max_length)
            
            prompt = f"""
            Please provide a comprehensive summary of the following content. 
            The summary should:
            1. Identify all the different ideas or concepts in the document
            2. Give the gist of each concept
            3. Be concise but complete
            4. Not exceed {max_length} words
            
            Content:
            {context}
            
            Summary:
            """
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_length,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating summary with LLM: {e}")
            # Fallback to extractive summary
            return self._extractive_summary(context, max_length)
    
    def _extractive_summary(self, text: str, max_length: int = 500) -> str:
        """Fallback extractive summary when LLM is not available"""
        sentences = text.split('. ')
        if len(sentences) <= 3:
            return text[:max_length]
        
        # Simple extractive summary - take first few sentences
        summary_sentences = sentences[:min(3, len(sentences))]
        summary = '. '.join(summary_sentences)
        
        if not summary.endswith('.'):
            summary += '.'
        
        return summary[:max_length]
    
    def generate_followup_questions(self, summary: str, num_questions: int = 3) -> List[str]:
        """Generate follow-up questions based on the summary"""
        try:
            if not self.api_key:
                return self._default_followup_questions()
            
            prompt = f"""
            Based on the following summary, generate {num_questions} relevant follow-up questions 
            that would help someone learn more about the topic. Make the questions specific and useful.
            
            Summary:
            {summary}
            
            Follow-up questions:
            """
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7
            )
            
            questions_text = response.choices[0].message.content.strip()
            questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
            
            return questions[:num_questions]
            
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {e}")
            return self._default_followup_questions()
    
    def _default_followup_questions(self) -> List[str]:
        """Default follow-up questions when LLM is not available"""
        return [
            "What are the main topics covered in this document?",
            "What are the key insights or conclusions?",
            "What additional information would be helpful?"
        ]

class SummarizationPlugin:
    """Main summarization plugin using k-means clustering approach"""
    
    def __init__(self, num_clustering_workers: int = 2, num_llm_workers: int = 2):
        self.num_clustering_workers = num_clustering_workers
        self.num_llm_workers = num_llm_workers
        
        # Initialize Ray workers
        self.clustering_workers = [ClusteringWorker.remote() for _ in range(num_clustering_workers)]
        self.llm_workers = [LLMWorker.remote() for _ in range(num_llm_workers)]
        
        logger.info(f"SummarizationPlugin initialized with {num_clustering_workers} clustering workers and {num_llm_workers} LLM workers")
    
    def summarize_single_file(self, chunks: List[TextChunk], max_summary_length: int = 500) -> SummaryResult:
        """Summarize a single file using k-means clustering approach"""
        logger.info(f"Summarizing file with {len(chunks)} chunks")
        
        if not chunks:
            return SummaryResult(
                summary="No content to summarize.",
                topics=[],
                chunk_sources=[],
                confidence_score=0.0
            )
        
        # Step 1: Extract embeddings for clustering
        embeddings = np.array([chunk.embedding for chunk in chunks])
        
        # Step 2: Perform k-means clustering
        cluster_labels = self._cluster_embeddings(embeddings)
        
        # Step 3: Select representative chunks from each cluster
        representative_chunks = self._select_representative_chunks(chunks, cluster_labels)
        
        # Step 4: Generate summary from representative chunks
        summary_result = self._generate_summary_from_chunks(representative_chunks, max_summary_length)
        
        logger.info(f"Generated summary with {len(representative_chunks)} representative chunks")
        return summary_result
    
    def summarize_multiple_files(self, file_chunks: Dict[str, List[TextChunk]], 
                               max_summary_length: int = 800) -> SummaryResult:
        """Summarize multiple files using combined approach"""
        logger.info(f"Summarizing {len(file_chunks)} files")
        
        # Combine all chunks from all files
        all_chunks = []
        for file_id, chunks in file_chunks.items():
            for chunk in chunks:
                chunk.file_id = file_id  # Ensure file_id is set
                all_chunks.append(chunk)
        
        if not all_chunks:
            return SummaryResult(
                summary="No content to summarize.",
                topics=[],
                chunk_sources=[],
                confidence_score=0.0
            )
        
        # Use same approach as single file but with more clusters for multiple files
        embeddings = np.array([chunk.embedding for chunk in all_chunks])
        
        # Increase number of clusters for multiple files
        n_clusters = min(max(5, len(all_chunks) // 8), 25)
        cluster_labels = self._cluster_embeddings(embeddings, n_clusters)
        
        # Select representative chunks
        representative_chunks = self._select_representative_chunks(all_chunks, cluster_labels)
        
        # Generate summary
        summary_result = self._generate_summary_from_chunks(representative_chunks, max_summary_length)
        
        logger.info(f"Generated multi-file summary with {len(representative_chunks)} representative chunks")
        return summary_result
    
    def _cluster_embeddings(self, embeddings: np.ndarray, n_clusters: int = None) -> np.ndarray:
        """Perform k-means clustering on embeddings"""
        # Use the first available clustering worker
        worker = self.clustering_workers[0]
        cluster_labels = ray.get(worker.cluster_chunks.remote(embeddings, n_clusters))
        return cluster_labels
    
    def _select_representative_chunks(self, chunks: List[TextChunk], 
                                    cluster_labels: np.ndarray) -> List[TextChunk]:
        """Select representative chunks from each cluster"""
        unique_clusters = np.unique(cluster_labels)
        representative_chunks = []
        
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_chunks = [chunk for i, chunk in enumerate(chunks) if cluster_mask[i]]
            
            if not cluster_chunks:
                continue
            
            # Select representative chunk from this cluster
            # Priority: first chunk chronologically, then most central to cluster
            representative_chunk = self._select_best_representative(chunk, cluster_chunks)
            representative_chunks.append(representative_chunk)
        
        # Sort by original position to maintain document flow
        representative_chunks.sort(key=lambda x: (x.file_id, x.position))
        
        return representative_chunks
    
    def _select_best_representative(self, chunks: List[TextChunk]) -> TextChunk:
        """Select the best representative chunk from a cluster"""
        if len(chunks) == 1:
            return chunks[0]
        
        # Priority 1: First two chunks chronologically get top priority
        if chunks[0].position <= 1:
            return chunks[0]
        
        # Priority 2: Select chunk with highest average similarity to other chunks in cluster
        if len(chunks) > 1:
            similarities = []
            for i, chunk in enumerate(chunks):
                similarity_sum = 0
                for j, other_chunk in enumerate(chunks):
                    if i != j:
                        similarity = np.dot(chunk.embedding, other_chunk.embedding) / (
                            np.linalg.norm(chunk.embedding) * np.linalg.norm(other_chunk.embedding)
                        )
                        similarity_sum += similarity
                
                avg_similarity = similarity_sum / (len(chunks) - 1)
                similarities.append((chunk, avg_similarity))
            
            # Return chunk with highest average similarity
            best_chunk, _ = max(similarities, key=lambda x: x[1])
            return best_chunk
        
        return chunks[0]
    
    def _generate_summary_from_chunks(self, chunks: List[TextChunk], 
                                    max_length: int) -> SummaryResult:
        """Generate summary from representative chunks"""
        if not chunks:
            return SummaryResult(
                summary="No content to summarize.",
                topics=[],
                chunk_sources=[],
                confidence_score=0.0
            )
        
        # Combine chunks into context
        context_parts = []
        for chunk in chunks:
            context_parts.append(f"[{chunk.file_id}:{chunk.position}] {chunk.text}")
        
        context = "\n\n".join(context_parts)
        
        # Generate summary using LLM worker
        worker = self.llm_workers[0]
        summary = ray.get(worker.generate_summary.remote(context, max_length))
        
        # Generate follow-up questions
        followup_questions = ray.get(worker.generate_followup_questions.remote(summary))
        
        # Calculate confidence score based on chunk diversity
        confidence_score = self._calculate_confidence_score(chunks)
        
        # Extract topics from summary (simple keyword extraction)
        topics = self._extract_topics(summary)
        
        return SummaryResult(
            summary=summary,
            topics=topics,
            chunk_sources=chunks,
            confidence_score=confidence_score,
            metadata={
                "num_chunks": len(chunks),
                "followup_questions": followup_questions,
                "context_length": len(context)
            }
        )
    
    def _calculate_confidence_score(self, chunks: List[TextChunk]) -> float:
        """Calculate confidence score based on chunk diversity and coverage"""
        if len(chunks) <= 1:
            return 0.5
        
        # Calculate diversity score based on embedding distances
        embeddings = np.array([chunk.embedding for chunk in chunks])
        distances = []
        
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                distance = np.linalg.norm(embeddings[i] - embeddings[j])
                distances.append(distance)
        
        avg_distance = np.mean(distances) if distances else 0
        max_possible_distance = 2.0  # For normalized embeddings
        
        # Normalize to 0-1 range
        diversity_score = min(avg_distance / max_possible_distance, 1.0)
        
        # Combine with coverage score (number of chunks)
        coverage_score = min(len(chunks) / 10.0, 1.0)
        
        # Final confidence is average of diversity and coverage
        confidence = (diversity_score + coverage_score) / 2.0
        
        return confidence
    
    def _extract_topics(self, summary: str) -> List[str]:
        """Extract topics from summary (simple keyword extraction)"""
        # Simple topic extraction - look for capitalized words and key phrases
        words = summary.lower().split()
        
        # Common topic indicators
        topic_indicators = [
            'machine learning', 'artificial intelligence', 'data science',
            'neural networks', 'deep learning', 'natural language processing',
            'computer vision', 'robotics', 'automation', 'algorithm',
            'model', 'training', 'prediction', 'analysis'
        ]
        
        topics = []
        for indicator in topic_indicators:
            if indicator in summary.lower():
                topics.append(indicator.title())
        
        # If no specific topics found, extract first few capitalized words
        if not topics:
            capitalized_words = [word for word in words if word[0].isupper() and len(word) > 3]
            topics = capitalized_words[:3]
        
        return topics[:5]  # Return top 5 topics

if __name__ == "__main__":
    # Initialize Ray
    ray.init(address='ray://ray-head-service:10001', ignore_reinit_error=True)
    
    # Test the summarization plugin
    summarization_plugin = SummarizationPlugin()
    
    # Create sample chunks (normally from embeddings_plugin)
    sample_chunks = [
        TextChunk(
            text="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            embedding=np.random.rand(384),  # Mock embedding
            chunk_id="chunk_1",
            file_id="sample_doc",
            position=0
        ),
        TextChunk(
            text="Deep learning uses neural networks with multiple layers to process complex data patterns.",
            embedding=np.random.rand(384),
            chunk_id="chunk_2", 
            file_id="sample_doc",
            position=1
        ),
        TextChunk(
            text="Natural language processing enables computers to understand and generate human language.",
            embedding=np.random.rand(384),
            chunk_id="chunk_3",
            file_id="sample_doc", 
            position=2
        )
    ]
    
    # Test summarization
    result = summarization_plugin.summarize_single_file(sample_chunks)
    
    print("Summary Result:")
    print(f"Summary: {result.summary}")
    print(f"Topics: {result.topics}")
    print(f"Confidence Score: {result.confidence_score:.2f}")
    print(f"Number of source chunks: {len(result.chunk_sources)}")
    
    ray.shutdown()
