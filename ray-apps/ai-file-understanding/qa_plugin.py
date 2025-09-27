#!/usr/bin/env python3
"""
Ray-based Q&A Plugin
Implements semantic search and power law dynamics for intelligent Q&A
"""

import ray
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import openai
import os
from embeddings_plugin import TextChunk, EmbeddingsPlugin

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QAResult:
    """Result of Q&A process"""
    answer: str
    confidence_score: float
    source_chunks: List[TextChunk]
    followup_questions: List[str]
    query_type: str  # 'direct' or 'broad'
    metadata: Dict[str, Any] = None

@ray.remote
class SemanticSearchWorker:
    """Ray remote worker for semantic search operations"""
    
    def __init__(self):
        logger.info("SemanticSearchWorker initialized")
    
    def calculate_relevance_scores(self, query_embedding: np.ndarray, 
                                 chunks: List[TextChunk]) -> List[Tuple[TextChunk, float]]:
        """Calculate relevance scores for chunks given a query"""
        similarities = []
        
        for chunk in chunks:
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, chunk.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk.embedding)
            )
            similarities.append((chunk, float(similarity)))
        
        # Sort by relevance score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities
    
    def apply_power_law_filtering(self, similarities: List[Tuple[TextChunk, float]], 
                                min_chunks: int = 5, max_chunks: int = 50) -> List[Tuple[TextChunk, float]]:
        """Apply power law dynamics to filter relevant chunks"""
        if len(similarities) <= min_chunks:
            return similarities
        
        # Take top 50 chunks for analysis
        top_chunks = similarities[:50]
        
        if len(top_chunks) < 2:
            return similarities[:min_chunks]
        
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
        
        # Ensure we have at least min_chunks and at most max_chunks
        if len(filtered_chunks) < min_chunks:
            filtered_chunks = top_chunks[:min_chunks]
        elif len(filtered_chunks) > max_chunks:
            filtered_chunks = filtered_chunks[:max_chunks]
        
        # Determine query type based on score distribution
        query_type = self._determine_query_type(scores)
        
        logger.info(f"Filtered {len(similarities)} chunks to {len(filtered_chunks)} using power law dynamics")
        logger.info(f"Query type: {query_type}")
        
        return filtered_chunks, query_type
    
    def _determine_query_type(self, scores: List[float]) -> str:
        """Determine if query is direct or broad based on score distribution"""
        if len(scores) < 3:
            return 'direct'
        
        # Calculate the steepness of the power law curve
        top_10_percent = int(len(scores) * 0.1)
        bottom_10_percent = int(len(scores) * 0.1)
        
        if top_10_percent == 0 or bottom_10_percent == 0:
            return 'direct'
        
        top_scores = scores[:top_10_percent]
        bottom_scores = scores[-bottom_10_percent:]
        
        top_avg = np.mean(top_scores)
        bottom_avg = np.mean(bottom_scores)
        
        # If there's a steep drop-off, it's a direct question
        # If scores are more evenly distributed, it's a broad question
        steepness = (top_avg - bottom_avg) / top_avg
        
        return 'direct' if steepness > 0.3 else 'broad'

@ray.remote
class LLMQAWorker:
    """Ray remote worker for LLM-based Q&A operations"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        
        if self.api_key:
            openai.api_key = self.api_key
        
        logger.info(f"LLMQAWorker initialized with model: {model}")
    
    def generate_answer(self, query: str, context_chunks: List[TextChunk], 
                       query_type: str) -> Tuple[str, float]:
        """Generate answer using LLM with context from relevant chunks"""
        try:
            # Prepare context from chunks
            context = self._prepare_context(context_chunks, query_type)
            
            if not self.api_key:
                # Fallback to simple retrieval-based answer
                return self._simple_answer(query, context_chunks)
            
            # Create prompt based on query type
            if query_type == 'direct':
                prompt = self._create_direct_prompt(query, context)
            else:
                prompt = self._create_broad_prompt(query, context)
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content.strip()
            confidence = self._calculate_confidence(context_chunks, query_type)
            
            return answer, confidence
            
        except Exception as e:
            logger.error(f"Error generating answer with LLM: {e}")
            return self._simple_answer(query, context_chunks)
    
    def _prepare_context(self, chunks: List[TextChunk], query_type: str) -> str:
        """Prepare context string from chunks"""
        if query_type == 'direct':
            # For direct questions, focus on most relevant chunks
            context_parts = []
            for chunk in chunks[:10]:  # Limit for direct questions
                context_parts.append(f"[{chunk.file_id}:{chunk.position}] {chunk.text}")
        else:
            # For broad questions, include more context
            context_parts = []
            for chunk in chunks[:25]:  # More context for broad questions
                context_parts.append(f"[{chunk.file_id}:{chunk.position}] {chunk.text}")
        
        return "\n\n".join(context_parts)
    
    def _create_direct_prompt(self, query: str, context: str) -> str:
        """Create prompt for direct questions"""
        return f"""
        Based on the provided context, please answer the following question directly and concisely.
        If the answer is not found in the context, please say so.
        
        Question: {query}
        
        Context:
        {context}
        
        Answer:
        """
    
    def _create_broad_prompt(self, query: str, context: str) -> str:
        """Create prompt for broad questions"""
        return f"""
        Based on the provided context, please provide a comprehensive answer to the following question.
        Include relevant details and explanations to give a complete understanding of the topic.
        
        Question: {query}
        
        Context:
        {context}
        
        Comprehensive Answer:
        """
    
    def _calculate_confidence(self, chunks: List[TextChunk], query_type: str) -> float:
        """Calculate confidence score based on chunk relevance and coverage"""
        if not chunks:
            return 0.0
        
        # Base confidence on number of relevant chunks
        chunk_count_score = min(len(chunks) / 10.0, 1.0)
        
        # Adjust for query type
        if query_type == 'direct':
            # Direct questions benefit from fewer, more relevant chunks
            confidence = chunk_count_score * 0.8 + 0.2
        else:
            # Broad questions benefit from more comprehensive coverage
            confidence = chunk_count_score * 0.6 + 0.4
        
        return min(confidence, 1.0)
    
    def _simple_answer(self, query: str, chunks: List[TextChunk]) -> Tuple[str, float]:
        """Fallback simple answer when LLM is not available"""
        if not chunks:
            return "I couldn't find relevant information to answer your question.", 0.0
        
        # Simple extractive answer - return the most relevant chunk
        answer = chunks[0].text[:300] + "..." if len(chunks[0].text) > 300 else chunks[0].text
        confidence = 0.5
        
        return answer, confidence
    
    def generate_followup_questions(self, query: str, answer: str, 
                                  context_chunks: List[TextChunk]) -> List[str]:
        """Generate relevant follow-up questions"""
        try:
            if not self.api_key:
                return self._default_followup_questions(query)
            
            # Extract topics from context
            topics = set()
            for chunk in context_chunks[:5]:
                words = chunk.text.lower().split()
                for word in words:
                    if len(word) > 5 and word.isalpha():
                        topics.add(word)
            
            topics_str = ", ".join(list(topics)[:5])
            
            prompt = f"""
            Based on the original question and answer below, generate 3 relevant follow-up questions
            that would help someone learn more about this topic. The questions should be specific
            and build upon the information provided.
            
            Original Question: {query}
            Answer: {answer}
            Related Topics: {topics_str}
            
            Follow-up Questions:
            """
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7
            )
            
            questions_text = response.choices[0].message.content.strip()
            questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
            
            return questions[:3]
            
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {e}")
            return self._default_followup_questions(query)
    
    def _default_followup_questions(self, query: str) -> List[str]:
        """Default follow-up questions when LLM is not available"""
        return [
            f"Can you provide more details about {query}?",
            "What are the key points to remember?",
            "Are there any related topics I should know about?"
        ]

class QAPlugin:
    """Main Q&A plugin with semantic search and power law dynamics"""
    
    def __init__(self, embeddings_plugin: EmbeddingsPlugin, 
                 num_search_workers: int = 2, num_llm_workers: int = 2):
        self.embeddings_plugin = embeddings_plugin
        self.num_search_workers = num_search_workers
        self.num_llm_workers = num_llm_workers
        
        # Initialize Ray workers
        self.search_workers = [SemanticSearchWorker.remote() for _ in range(num_search_workers)]
        self.llm_workers = [LLMQAWorker.remote() for _ in range(num_llm_workers)]
        
        logger.info(f"QAPlugin initialized with {num_search_workers} search workers and {num_llm_workers} LLM workers")
    
    def answer_question(self, query: str, chunks: List[TextChunk], 
                       min_chunks: int = 5, max_chunks: int = 50) -> QAResult:
        """Answer a question using semantic search and power law dynamics"""
        logger.info(f"Processing question: {query[:100]}...")
        
        if not chunks:
            return QAResult(
                answer="I don't have any content to search through to answer your question.",
                confidence_score=0.0,
                source_chunks=[],
                followup_questions=[],
                query_type="unknown"
            )
        
        # Step 1: Compute query embedding
        query_embedding = self.embeddings_plugin.compute_query_embedding(query)
        
        # Step 2: Calculate relevance scores using semantic search
        search_worker = self.search_workers[0]
        similarities = ray.get(search_worker.calculate_relevance_scores.remote(query_embedding, chunks))
        
        # Step 3: Apply power law filtering to select relevant chunks
        filtered_chunks, query_type = ray.get(
            search_worker.apply_power_law_filtering.remote(similarities, min_chunks, max_chunks)
        )
        
        relevant_chunks = [chunk for chunk, score in filtered_chunks]
        
        # Step 4: Generate answer using LLM
        llm_worker = self.llm_workers[0]
        answer, confidence = ray.get(
            llm_worker.generate_answer.remote(query, relevant_chunks, query_type)
        )
        
        # Step 5: Generate follow-up questions
        followup_questions = ray.get(
            llm_worker.generate_followup_questions.remote(query, answer, relevant_chunks)
        )
        
        logger.info(f"Generated answer with {len(relevant_chunks)} source chunks, confidence: {confidence:.2f}")
        
        return QAResult(
            answer=answer,
            confidence_score=confidence,
            source_chunks=relevant_chunks,
            followup_questions=followup_questions,
            query_type=query_type,
            metadata={
                "total_chunks_searched": len(chunks),
                "relevant_chunks_used": len(relevant_chunks),
                "query_embedding_computed": True
            }
        )
    
    def answer_question_multiple_files(self, query: str, 
                                     file_chunks: Dict[str, List[TextChunk]],
                                     min_chunks: int = 5, max_chunks: int = 50) -> QAResult:
        """Answer a question across multiple files"""
        logger.info(f"Processing question across {len(file_chunks)} files: {query[:100]}...")
        
        # Combine all chunks from all files
        all_chunks = []
        for file_id, chunks in file_chunks.items():
            for chunk in chunks:
                chunk.file_id = file_id  # Ensure file_id is set
                all_chunks.append(chunk)
        
        # Use same approach as single file but with adjusted parameters for multiple files
        return self.answer_question(query, all_chunks, min_chunks, max_chunks)
    
    def batch_answer_questions(self, queries: List[str], chunks: List[TextChunk]) -> List[QAResult]:
        """Answer multiple questions in parallel"""
        logger.info(f"Processing {len(queries)} questions in parallel")
        
        # Distribute questions across LLM workers
        futures = []
        for i, query in enumerate(queries):
            worker = self.llm_workers[i % len(self.llm_workers)]
            # Create a future for each question
            future = ray.put(self.answer_question(query, chunks))
            futures.append(future)
        
        # Collect results
        results = ray.get(futures)
        
        logger.info(f"Completed batch processing of {len(queries)} questions")
        return results

if __name__ == "__main__":
    # Initialize Ray
    ray.init(address='ray://ray-head-service:10001', ignore_reinit_error=True)
    
    # Test the Q&A plugin
    embeddings_plugin = EmbeddingsPlugin()
    qa_plugin = QAPlugin(embeddings_plugin)
    
    # Create sample chunks (normally from embeddings_plugin)
    sample_chunks = [
        TextChunk(
            text="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data without being explicitly programmed.",
            embedding=np.random.rand(384),  # Mock embedding
            chunk_id="chunk_1",
            file_id="sample_doc",
            position=0
        ),
        TextChunk(
            text="Deep learning uses neural networks with multiple layers to process complex data patterns and make predictions.",
            embedding=np.random.rand(384),
            chunk_id="chunk_2", 
            file_id="sample_doc",
            position=1
        ),
        TextChunk(
            text="Natural language processing enables computers to understand and generate human language for various applications.",
            embedding=np.random.rand(384),
            chunk_id="chunk_3",
            file_id="sample_doc", 
            position=2
        )
    ]
    
    # Test Q&A
    queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "What are the applications of AI?"
    ]
    
    for query in queries:
        result = qa_plugin.answer_question(query, sample_chunks)
        
        print(f"\nQuestion: {query}")
        print(f"Answer: {result.answer}")
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"Query Type: {result.query_type}")
        print(f"Source Chunks: {len(result.source_chunks)}")
        print(f"Follow-up Questions: {result.followup_questions}")
    
    ray.shutdown()
