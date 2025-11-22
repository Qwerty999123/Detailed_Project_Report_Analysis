import hashlib
from typing import List, Dict, Optional
from dataclasses import dataclass
import re

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

@dataclass
class DocumentChunk:
    """Structure for document chunks with metadata"""
    content: str
    metadata: Dict
    embedding: Optional[List[float]] = None


class IntelligentDocumentChunker:
    """Advanced document chunking with multiple strategies"""
    
    def __init__(self):
        self.semantic_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.structure_patterns = {
            'section': r'^#{1,3}\s+(.+)$',
            'subsection': r'^#{4,6}\s+(.+)$',
            'list_item': r'^[-\*\+]\s+(.+)$',
            'numbered_item': r'^\d+\.\s+(.+)$',
            'table_row': r'\|.+\|'
        }
    
    def chunk_document(self, documents: List[Document]) -> List[DocumentChunk]:
        """Apply intelligent chunking strategies"""
        all_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            # Strategy 1: Semantic chunking
            semantic_chunks = self.semantic_chunking(doc, doc_idx)
            
            # Strategy 2: Structural chunking
            structural_chunks = self.structural_chunking(doc, doc_idx)
            
            # Strategy 3: Overlapping windows for continuity
            overlap_chunks = self.create_overlap_chunks(doc, doc_idx)
            
            # Combine strategies and deduplicate
            combined_chunks = self.merge_chunk_strategies(
                semantic_chunks, structural_chunks, overlap_chunks
            )
            
            all_chunks.extend(combined_chunks)
        
        return all_chunks
    
    def semantic_chunking(self, doc: Document, doc_idx: int) -> List[DocumentChunk]:
        """Semantic-based chunking"""
        chunks = self.semantic_splitter.split_text(doc.page_content)
        
        chunk_objects = []
        for chunk_idx, chunk_text in enumerate(chunks):
            chunk_obj = DocumentChunk(
                content=chunk_text,
                metadata={
                    'doc_index': doc_idx,
                    'chunk_index': chunk_idx,
                    'chunk_type': 'semantic',
                    'section': self.identify_section(chunk_text),
                    'page': doc.metadata.get('page', doc_idx),
                    'source': doc.metadata.get('source', ''),
                    'content_type': self.classify_content_type(chunk_text)
                }
            )
            chunk_objects.append(chunk_obj)
        
        return chunk_objects
    
    def structural_chunking(self, doc: Document, doc_idx: int) -> List[DocumentChunk]:
        """Structure-aware chunking"""
        lines = doc.page_content.split('\n')
        current_section = "Introduction"
        current_chunk = ""
        chunk_objects = []
        chunk_idx = 0
        
        for line in lines:
            # Check for section headers
            if re.match(self.structure_patterns['section'], line):
                if current_chunk.strip():
                    chunk_obj = DocumentChunk(
                        content=current_chunk,
                        metadata={
                            'doc_index': doc_idx,
                            'chunk_index': chunk_idx,
                            'chunk_type': 'structural',
                            'section': current_section,
                            'page': doc.metadata.get('page', doc_idx),
                            'source': doc.metadata.get('source', ''),
                            'is_section_complete': True
                        }
                    )
                    chunk_objects.append(chunk_obj)
                    chunk_idx += 1
                
                current_section = line.strip()
                current_chunk = line + '\n'
            else:
                current_chunk += line + '\n'
                
                # Split if chunk gets too large
                if len(current_chunk) > 2500:
                    chunk_obj = DocumentChunk(
                        content=current_chunk,
                        metadata={
                            'doc_index': doc_idx,
                            'chunk_index': chunk_idx,
                            'chunk_type': 'structural',
                            'section': current_section,
                            'page': doc.metadata.get('page', doc_idx),
                            'source': doc.metadata.get('source', ''),
                            'is_section_complete': False
                        }
                    )
                    chunk_objects.append(chunk_obj)
                    chunk_idx += 1
                    current_chunk = ""
        
        # Add final chunk
        if current_chunk.strip():
            chunk_obj = DocumentChunk(
                content=current_chunk,
                metadata={
                    'doc_index': doc_idx,
                    'chunk_index': chunk_idx,
                    'chunk_type': 'structural',
                    'section': current_section,
                    'page': doc.metadata.get('page', doc_idx),
                    'source': doc.metadata.get('source', ''),
                    'is_section_complete': True
                }
            )
            chunk_objects.append(chunk_obj)
        
        return chunk_objects
    
    def create_overlap_chunks(self, doc: Document, doc_idx: int) -> List[DocumentChunk]:
        """Create overlapping chunks for continuity"""
        text = doc.page_content
        chunk_size = 2000
        overlap = 500
        chunks = []
        chunk_idx = 0
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk_text = text[i:i + chunk_size]
            if chunk_text.strip():
                chunk_obj = DocumentChunk(
                    content=chunk_text,
                    metadata={
                        'doc_index': doc_idx,
                        'chunk_index': chunk_idx,
                        'chunk_type': 'overlap',
                        'section': self.identify_section(chunk_text),
                        'page': doc.metadata.get('page', doc_idx),
                        'source': doc.metadata.get('source', ''),
                        'start_pos': i,
                        'end_pos': i + len(chunk_text)
                    }
                )
                chunks.append(chunk_obj)
                chunk_idx += 1
        
        return chunks
    
    def merge_chunk_strategies(self, *chunk_lists) -> List[DocumentChunk]:
        """Merge and deduplicate chunks from different strategies"""
        all_chunks = []
        for chunk_list in chunk_lists:
            all_chunks.extend(chunk_list)
        
        # Simple deduplication based on content similarity
        unique_chunks = []
        seen_hashes = set()
        
        for chunk in all_chunks:
            content_hash = hashlib.md5(chunk.content.encode()).hexdigest()
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def identify_section(self, text: str) -> str:
        """Identify the section type from text content"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['objective', 'goal', 'aim', 'purpose']):
            return 'objectives'
        elif any(word in text_lower for word in ['scope', 'coverage', 'boundary']):
            return 'scope'
        elif any(word in text_lower for word in ['requirement', 'specification', 'shall', 'must']):
            return 'requirements'
        elif any(word in text_lower for word in ['cost', 'budget', 'financial', 'estimate']):
            return 'financial'
        elif any(word in text_lower for word in ['implementation', 'execution', 'timeline']):
            return 'implementation'
        elif any(word in text_lower for word in ['infrastructure', 'hardware', 'software', 'application']):
            return 'technical'
        else:
            return 'general'
    
    def classify_content_type(self, text: str) -> str:
        """Classify the type of content"""
        if '|' in text and text.count('|') > 3:
            return 'table'
        elif re.search(r'^\d+\.', text.strip(), re.MULTILINE):
            return 'numbered_list'
        elif re.search(r'^[-\*\+]', text.strip(), re.MULTILINE):
            return 'bullet_list'
        elif len(text) > 1000:
            return 'paragraph'
        else:
            return 'text'