#!/usr/bin/env python3
"""
Ray-based File Processing Pipeline
Handles multiple file types with distributed processing
"""

import ray
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import os
import mimetypes
import subprocess
import tempfile
from pathlib import Path
import json
import hashlib
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FileMetadata:
    """Metadata for processed files"""
    file_id: str
    original_path: str
    file_type: str
    mime_type: str
    size: int
    created_at: str
    modified_at: str
    checksum: str
    processing_status: str = "pending"
    error_message: Optional[str] = None

@dataclass
class ProcessingResult:
    """Result of file processing"""
    file_id: str
    text_content: str
    metadata: FileMetadata
    processing_time: float
    success: bool
    intermediate_files: List[str] = None

class FileProcessor(ABC):
    """Abstract base class for file processors"""
    
    @abstractmethod
    def can_process(self, file_path: str, mime_type: str) -> bool:
        """Check if this processor can handle the file"""
        pass
    
    @abstractmethod
    def process(self, file_path: str, metadata: FileMetadata) -> ProcessingResult:
        """Process the file and extract text content"""
        pass

class TextProcessor(FileProcessor):
    """Processor for plain text files"""
    
    def can_process(self, file_path: str, mime_type: str) -> bool:
        return mime_type.startswith('text/')
    
    def process(self, file_path: str, metadata: FileMetadata) -> ProcessingResult:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return ProcessingResult(
                file_id=metadata.file_id,
                text_content=content,
                metadata=metadata,
                processing_time=0.0,
                success=True
            )
        except Exception as e:
            return ProcessingResult(
                file_id=metadata.file_id,
                text_content="",
                metadata=metadata,
                processing_time=0.0,
                success=False
            )

class PDFProcessor(FileProcessor):
    """Processor for PDF files"""
    
    def can_process(self, file_path: str, mime_type: str) -> bool:
        return mime_type == 'application/pdf'
    
    def process(self, file_path: str, metadata: FileMetadata) -> ProcessingResult:
        try:
            # Use pdftotext if available
            result = subprocess.run(
                ['pdftotext', file_path, '-'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                content = result.stdout
            else:
                # Fallback to PyPDF2 or similar
                content = self._fallback_pdf_extraction(file_path)
            
            return ProcessingResult(
                file_id=metadata.file_id,
                text_content=content,
                metadata=metadata,
                processing_time=0.0,
                success=True
            )
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            return ProcessingResult(
                file_id=metadata.file_id,
                text_content="",
                metadata=metadata,
                processing_time=0.0,
                success=False,
                error_message=str(e)
            )
    
    def _fallback_pdf_extraction(self, file_path: str) -> str:
        """Fallback PDF text extraction"""
        try:
            import PyPDF2
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except ImportError:
            return "PDF processing requires pdftotext or PyPDF2"

class VideoProcessor(FileProcessor):
    """Processor for video files - extracts audio and transcribes"""
    
    def can_process(self, file_path: str, mime_type: str) -> bool:
        return mime_type.startswith('video/')
    
    def process(self, file_path: str, metadata: FileMetadata) -> ProcessingResult:
        try:
            # Step 1: Extract audio from video
            audio_file = self._extract_audio(file_path)
            
            # Step 2: Transcribe audio to text
            transcript = self._transcribe_audio(audio_file)
            
            # Clean up temporary audio file
            if audio_file and os.path.exists(audio_file):
                os.remove(audio_file)
            
            return ProcessingResult(
                file_id=metadata.file_id,
                text_content=transcript,
                metadata=metadata,
                processing_time=0.0,
                success=True,
                intermediate_files=[audio_file] if audio_file else []
            )
        except Exception as e:
            logger.error(f"Error processing video {file_path}: {e}")
            return ProcessingResult(
                file_id=metadata.file_id,
                text_content="",
                metadata=metadata,
                processing_time=0.0,
                success=False,
                error_message=str(e)
            )
    
    def _extract_audio(self, video_file: str) -> str:
        """Extract audio from video using ffmpeg"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                result = subprocess.run([
                    'ffmpeg', '-i', video_file, '-vn', '-acodec', 'pcm_s16le', 
                    '-ar', '16000', '-ac', '1', temp_audio.name, '-y'
                ], capture_output=True, timeout=60)
                
                if result.returncode == 0:
                    return temp_audio.name
                else:
                    logger.error(f"FFmpeg error: {result.stderr}")
                    return None
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            return None
    
    def _transcribe_audio(self, audio_file: str) -> str:
        """Transcribe audio to text"""
        try:
            # Try using whisper if available
            result = subprocess.run([
                'whisper', audio_file, '--model', 'base', '--output_format', 'txt'
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                # Read the generated transcript
                transcript_file = audio_file.replace('.wav', '.txt')
                if os.path.exists(transcript_file):
                    with open(transcript_file, 'r') as f:
                        transcript = f.read()
                    os.remove(transcript_file)  # Clean up
                    return transcript
            
            # Fallback to simple message
            return "Video transcription requires whisper or similar tool"
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return "Audio transcription failed"

class AudioProcessor(FileProcessor):
    """Processor for audio files"""
    
    def can_process(self, file_path: str, mime_type: str) -> bool:
        return mime_type.startswith('audio/')
    
    def process(self, file_path: str, metadata: FileMetadata) -> ProcessingResult:
        try:
            # Use the same transcription logic as video processor
            transcript = self._transcribe_audio(file_path)
            
            return ProcessingResult(
                file_id=metadata.file_id,
                text_content=transcript,
                metadata=metadata,
                processing_time=0.0,
                success=True
            )
        except Exception as e:
            logger.error(f"Error processing audio {file_path}: {e}")
            return ProcessingResult(
                file_id=metadata.file_id,
                text_content="",
                metadata=metadata,
                processing_time=0.0,
                success=False,
                error_message=str(e)
            )
    
    def _transcribe_audio(self, audio_file: str) -> str:
        """Transcribe audio to text (same as video processor)"""
        try:
            result = subprocess.run([
                'whisper', audio_file, '--model', 'base', '--output_format', 'txt'
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                transcript_file = audio_file.replace(Path(audio_file).suffix, '.txt')
                if os.path.exists(transcript_file):
                    with open(transcript_file, 'r') as f:
                        transcript = f.read()
                    os.remove(transcript_file)
                    return transcript
            
            return "Audio transcription requires whisper or similar tool"
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return "Audio transcription failed"

class OfficeProcessor(FileProcessor):
    """Processor for Microsoft Office documents"""
    
    def can_process(self, file_path: str, mime_type: str) -> bool:
        office_types = [
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            'application/msword',
            'application/vnd.ms-excel',
            'application/vnd.ms-powerpoint'
        ]
        return mime_type in office_types
    
    def process(self, file_path: str, metadata: FileMetadata) -> ProcessingResult:
        try:
            # Convert to text using pandoc or similar
            result = subprocess.run([
                'pandoc', file_path, '-t', 'plain'
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                content = result.stdout
            else:
                # Fallback to python-docx or similar
                content = self._fallback_office_extraction(file_path, metadata.file_type)
            
            return ProcessingResult(
                file_id=metadata.file_id,
                text_content=content,
                metadata=metadata,
                processing_time=0.0,
                success=True
            )
        except Exception as e:
            logger.error(f"Error processing Office document {file_path}: {e}")
            return ProcessingResult(
                file_id=metadata.file_id,
                text_content="",
                metadata=metadata,
                processing_time=0.0,
                success=False,
                error_message=str(e)
            )
    
    def _fallback_office_extraction(self, file_path: str, file_type: str) -> str:
        """Fallback Office document extraction"""
        try:
            if file_type == 'docx':
                from docx import Document
                doc = Document(file_path)
                return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            elif file_type == 'xlsx':
                import pandas as pd
                df = pd.read_excel(file_path)
                return df.to_string()
            else:
                return "Office document processing requires pandoc or python-docx"
        except ImportError:
            return "Office document processing requires additional libraries"

@ray.remote
class FileProcessingWorker:
    """Ray remote worker for file processing"""
    
    def __init__(self):
        self.processors = [
            TextProcessor(),
            PDFProcessor(),
            VideoProcessor(),
            AudioProcessor(),
            OfficeProcessor()
        ]
        logger.info("FileProcessingWorker initialized")
    
    def process_file(self, file_path: str, metadata: FileMetadata) -> ProcessingResult:
        """Process a single file"""
        mime_type = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
        
        # Find appropriate processor
        for processor in self.processors:
            if processor.can_process(file_path, mime_type):
                logger.info(f"Processing {file_path} with {processor.__class__.__name__}")
                return processor.process(file_path, metadata)
        
        # No processor found
        return ProcessingResult(
            file_id=metadata.file_id,
            text_content="",
            metadata=metadata,
            processing_time=0.0,
            success=False,
            error_message=f"No processor found for mime type: {mime_type}"
        )

class FileProcessingPipeline:
    """Main file processing pipeline for multi-format file handling"""
    
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.workers = [FileProcessingWorker.remote() for _ in range(num_workers)]
        self.processing_graph = self._build_processing_graph()
        
        logger.info(f"FileProcessingPipeline initialized with {num_workers} workers")
    
    def _build_processing_graph(self) -> Dict[str, List[str]]:
        """Build the processing graph showing possible conversions"""
        return {
            'text': ['txt', 'md', 'log'],
            'pdf': ['pdf'],
            'video': ['mp4', 'avi', 'mov', 'mkv'],
            'audio': ['mp3', 'wav', 'flac', 'm4a'],
            'office': ['docx', 'xlsx', 'pptx', 'doc', 'xls', 'ppt']
        }
    
    def create_file_metadata(self, file_path: str, file_id: str = None) -> FileMetadata:
        """Create metadata for a file"""
        if not file_id:
            file_id = hashlib.md5(file_path.encode()).hexdigest()
        
        stat = os.stat(file_path)
        mime_type = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
        
        return FileMetadata(
            file_id=file_id,
            original_path=file_path,
            file_type=Path(file_path).suffix.lower(),
            mime_type=mime_type,
            size=stat.st_size,
            created_at=str(stat.st_ctime),
            modified_at=str(stat.st_mtime),
            checksum=self._calculate_checksum(file_path)
        )
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate file checksum"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def process_single_file(self, file_path: str, file_id: str = None) -> ProcessingResult:
        """Process a single file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        metadata = self.create_file_metadata(file_path, file_id)
        worker = self.workers[0]  # Use first worker for single file
        
        return ray.get(worker.process_file.remote(file_path, metadata))
    
    def process_multiple_files(self, file_paths: List[str]) -> Dict[str, ProcessingResult]:
        """Process multiple files in parallel"""
        logger.info(f"Processing {len(file_paths)} files")
        
        # Create metadata for all files
        file_metadata = {}
        for file_path in file_paths:
            if os.path.exists(file_path):
                file_id = hashlib.md5(file_path.encode()).hexdigest()
                file_metadata[file_id] = self.create_file_metadata(file_path, file_id)
            else:
                logger.warning(f"File not found: {file_path}")
        
        # Distribute files across workers
        futures = []
        for i, (file_path, metadata) in enumerate(file_metadata.items()):
            worker = self.workers[i % len(self.workers)]
            future = worker.process_file.remote(file_path, metadata)
            futures.append((file_path, future))
        
        # Collect results
        results = {}
        for file_path, future in futures:
            result = ray.get(future)
            results[result.file_id] = result
        
        logger.info(f"Processed {len(results)} files successfully")
        return results
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get list of supported file formats"""
        return self.processing_graph
    
    def get_processing_status(self, results: Dict[str, ProcessingResult]) -> Dict[str, Any]:
        """Get processing status summary"""
        total = len(results)
        successful = sum(1 for r in results.values() if r.success)
        failed = total - successful
        
        return {
            "total_files": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0,
            "failed_files": [r.file_id for r in results.values() if not r.success]
        }

if __name__ == "__main__":
    # Initialize Ray
    ray.init(address='ray://ray-head-service:10001', ignore_reinit_error=True)
    
    # Test the file processing pipeline
    pipeline = FileProcessingPipeline()
    
    # Test with sample files (create dummy files for testing)
    test_files = []
    
    # Create a sample text file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a sample text file for testing the file processing pipeline.")
        test_files.append(f.name)
    
    # Process files
    if test_files:
        results = pipeline.process_multiple_files(test_files)
        
        print("Processing Results:")
        for file_id, result in results.items():
            print(f"File ID: {file_id}")
            print(f"Success: {result.success}")
            print(f"Content: {result.text_content[:100]}...")
            print(f"Metadata: {result.metadata}")
            print()
        
        # Get status
        status = pipeline.get_processing_status(results)
        print("Processing Status:", status)
        
        # Clean up test files
        for file_path in test_files:
            os.unlink(file_path)
    
    print("Supported formats:", pipeline.get_supported_formats())
    
    ray.shutdown()
