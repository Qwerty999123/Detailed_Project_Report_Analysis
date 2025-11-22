"""
Complete RAG-Enhanced Multi-Agent System for DPR-to-Requirements Generation
Handles large documents (100+ pages) efficiently using vector storage and retrieval
"""

import os
import math
from typing import List, Dict
from dataclasses import dataclass
from datetime import datetime

from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

from chunks import DocumentChunk, IntelligentDocumentChunker
from vector_store import RAGVectorStore
from document_parser_agent import DocumentParserAgent
from requirements_extracter_agent import RAGRequirementExtractorAgent
from tech_specialist_agent import TechnicalSpecialistAgent
from quality_reviewer_agent import QualityReviewerAgent
# from refine_requirements import RefineRequirements

from google import genai

CLIENT = genai.Client(
    api_key="AIzaSyDZlba6owD92R-PFb0CgkEpHUEdcehOTRg"
)



@dataclass
class ProcessingResult:
    """Structure for processing results"""
    requirements_document: str
    processing_metadata: Dict
    quality_metrics: Dict
    document_structure: Dict
    processing_time: float










class RAGDPRProcessor:
    """Main processor class that orchestrates the entire pipeline"""
    
    def __init__(self, embedding_model: str = "BAAI/bge-base-en-v1.5"):
        self.chunker = IntelligentDocumentChunker()
        self.vector_store = RAGVectorStore(embedding_model_name=embedding_model)
        
        # Initialize agents
        self.document_parser = DocumentParserAgent()
        self.requirement_extractor = None  # Will be initialized after vector store
        self.technical_specialist = None
        self.quality_reviewer = QualityReviewerAgent()
        
        self.processing_stats = {
            'start_time': None,
            'end_time': None,
            'total_chunks': 0,
            'requirements_extracted': 0,
            'quality_score': 0
        }
    
    def process_dpr_document(self, file_path: str, output_format: str = 'markdown') -> ProcessingResult:
        """Main processing pipeline for DPR documents"""
        
        self.processing_stats['start_time'] = datetime.now()
        
        try:
            # Step 1: Load and chunk document
            print("Loading and chunking document...")
            document_chunks = self.load_and_chunk_document(file_path)
            self.processing_stats['total_chunks'] = len(document_chunks)
            
            # Step 2: Create vector store
            print("Creating vector store...")
            self.vector_store.create_vector_store(document_chunks)
            
            # Initialize agents that need vector store
            self.requirement_extractor = RAGRequirementExtractorAgent(self.vector_store)
            self.technical_specialist = TechnicalSpecialistAgent(self.vector_store)
            
            # Step 3: Analyze document structure
            print("Analyzing document structure...")
            document_structure = self.document_parser.analyze_document_structure(document_chunks)
            
            # Step 4: Extract requirements progressively
            print("Extracting requirements...")
            requirements = self.extract_requirements_by_sections(document_structure, document_chunks)
            self.processing_stats['requirements_extracted'] = len(requirements)
            
            # Step 5: Technical enhancement
            print("Enhancing technical specifications...")
            enhanced_requirements = self.technical_specialist.enhance_technical_requirements(requirements)
            
            # Step 6: Quality review
            print("Performing quality review...")
            quality_report = self.quality_reviewer.review_requirements_quality(enhanced_requirements)
            self.processing_stats['quality_score'] = quality_report['overall_score']
            
            # Step 7: Generate final document
            print("Generating final document...")
            final_document = self.generate_requirements_document(
                quality_report['validated_requirements'], 
                document_structure, 
                output_format
            )
            
            self.processing_stats['end_time'] = datetime.now()
            processing_time = (self.processing_stats['end_time'] - self.processing_stats['start_time']).total_seconds()
            
            return ProcessingResult(
                requirements_document=final_document,
                processing_metadata=self.processing_stats.copy(),
                quality_metrics=quality_report,
                document_structure=document_structure,
                processing_time=processing_time
            )
            
        except Exception as e:
            print(f"Error processing document: {e}")
            raise
    
    def load_and_chunk_document(self, file_path: str) -> List[DocumentChunk]:
        """Load document and create intelligent chunks"""
        
        # Select appropriate loader
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        else:
            loader = TextLoader(file_path)
        
        # Load document
        documents = loader.load()
        
        # Create intelligent chunks
        return self.chunker.chunk_document(documents)
    
    # def _llm_call(self, document_chunks: List[DocumentChunk], section_name):

    
    def extract_requirements_by_sections(self, document_structure: Dict, document_chunks: List[DocumentChunk]) -> List[Dict]:
        """Extract requirements section by section"""
        all_requirements = []
        section_map = document_structure.get('section_map', {})
        print(section_map)
        
        # Process each major section
        priority_sections = ['requirements', 'objectives', 'scope', 'technical', 'implementation']
        max_chunks_per_call = 20
        total_calls = 0
        for section in section_map:
            print(f"Processing {section} section...")
            num_chunks = section_map[section]

            if num_chunks <= max_chunks_per_call:
                # Process entire section in one call
                print(f"  Making 1 API call for {num_chunks} chunks")

                relevant_chunks = []
                for chunk in document_chunks:
                    if chunk.metadata['section'] == section:
                        relevant_chunks.append(chunk)
                
                print(len(relevant_chunks))

                if relevant_chunks:
                    section_requirements = self.requirement_extractor.extract_requirements_from_chunks(
                        relevant_chunks, section
                    )
                    all_requirements.extend(section_requirements)

                
                total_calls += 1
                
            else:
                # Break section into groups
                num_groups = math.ceil(num_chunks / max_chunks_per_call)
                print(f"  Making {num_groups} API calls (groups of ~{max_chunks_per_call} chunks)")
                
                chunks = []
                for chunk in document_chunks:
                    if chunk.metadata['section'] == section:
                        chunks.append(chunk)

                print(len(chunks))

                for group_idx in range(num_groups):
                    start_idx = group_idx * max_chunks_per_call
                    end_idx = min(start_idx + max_chunks_per_call, num_chunks)
                    chunk_group = chunks[start_idx:end_idx]
                    
                    print(f"    Group {group_idx + 1}: chunks {start_idx + 1}-{end_idx}")
                    
                    if chunk_group:
                        section_requirements = self.requirement_extractor.extract_requirements_from_chunks(
                            chunk_group, section
                        )
                        all_requirements.extend(section_requirements)
                    
                    total_calls += 1
            
        print(f"\nTotal API calls made: {total_calls}")

        #     # Get chunks for this section
        #     section_query = f"{section} specifications requirements"
        #     relevant_chunks = self.vector_store.retrieve_relevant_chunks(section_query, k=10)
        #     print(relevant_chunks)
            
        #     if relevant_chunks:
        #         section_requirements = self.requirement_extractor.extract_requirements_from_chunks(
        #             relevant_chunks, section
        #         )
        #         all_requirements.extend(section_requirements)
        
        # # Process remaining sections
        # for section in section_map:
        #     if section not in priority_sections:
        #         print(f"Processing {section} section...")
        #         section_query = f"{section} requirements specifications"
        #         relevant_chunks = self.vector_store.retrieve_relevant_chunks(section_query, k=5)
        #         print(len(relevant_chunks))
                
        #         if relevant_chunks:
        #             section_requirements = self.requirement_extractor.extract_requirements_from_chunks(
        #                 relevant_chunks, section
        #             )
        #             all_requirements.extend(section_requirements)
        
        return all_requirements
    
    def generate_requirements_document(self, requirements: List[Dict], 
                                     document_structure: Dict, 
                                     output_format: str) -> str:
        """Generate final requirements document"""
        
        # Group requirements by type
        grouped_reqs = {
            'functional': [],
            'non-functional': [],
            'technical': [],
            'business': [],
            'other': []
        }
        
        for req in requirements:
            req_type = req.get('type', 'other')
            if req_type in grouped_reqs:
                grouped_reqs[req_type].append(req)
            else:
                grouped_reqs['other'].append(req)
        
        # Generate document content
        doc_content = []
        
        # Header
        # project_name = document_structure.get('document_metadata', {}).get('project_name', 'Unknown Project')
        # doc_content.append(f"# Requirements Document: {project_name}")
        # doc_content.append("")
        
        # # Document metadata
        # doc_content.append("## Document Information")
        # doc_content.append("")
        # metadata = document_structure.get('document_metadata', {})
        # for key, value in metadata.items():
        #     doc_content.append(f"- **{key.replace('_', ' ').title()}**: {value}")
        # doc_content.append("")
        
        # # Processing statistics
        # doc_content.append("## Processing Summary")
        # doc_content.append("")
        # doc_content.append(f"- **Total Requirements Extracted**: {len(requirements)}")
        # doc_content.append(f"- **Document Chunks Processed**: {self.processing_stats.get('total_chunks', 0)}")
        # doc_content.append(f"- **Quality Score**: {self.processing_stats.get('quality_score', 0):.1f}/10")
        # doc_content.append("")
        desc = []
        # Requirements by type
        for req_type, reqs in grouped_reqs.items():
            if reqs:
                doc_content.append(f"## {req_type.replace('_', '-').title()} Requirements")
                doc_content.append("")
                
                for req in reqs:

                    if req.get('description', 'No description') not in desc:

                        desc.append(req.get('description', 'No description'))

                        doc_content.append(f"### {req.get('id', 'REQ-UNKNOWN')}")
                        doc_content.append("")
                        doc_content.append(f"**Description**: {req.get('description', 'No description')}")
                        doc_content.append("")
            #             doc_content.append(f"**Priority**: {req.get('priority', 'unknown').title()}")
            #             doc_content.append("")
                        
            #             if req.get('technical_specifications'):
            #                 doc_content.append("**Technical Specifications**:")
            #                 for spec in req['technical_specifications']:
            #                     doc_content.append(f"- {spec}")
            #                 doc_content.append("")
                        
            #             if req.get('performance_criteria'):
            #                 doc_content.append("**Performance Criteria**:")
            #                 for criteria in req['performance_criteria']:
            #                     doc_content.append(f"- {criteria}")
            #                 doc_content.append("")
                        
            #             if req.get('acceptance_criteria'):
            #                 doc_content.append("**Acceptance Criteria**:")
            #                 for criteria in req['acceptance_criteria']:
            #                     doc_content.append(f"- {criteria}")
            #                 doc_content.append("")
                        
            #             source_chunk = req.get('source_chunk', {})
            #             if source_chunk:
            #                 doc_content.append(f"**Source**: {source_chunk.get('section', 'Unknown')} (Page {source_chunk.get('page', 'Unknown')})")
            #                 doc_content.append("")
                        
            #             doc_content.append("---")
            #             doc_content.append("")
        
        return '\n'.join(doc_content)


# Usage Examples
def example_usage():
    """Examples of how to use the system"""
    
    # Example 1: Simple processing
    processor = RAGDPRProcessor()
    
    filename = 'DPR-Pkg-X-021220.pdf'
    path = 'DPR'

    try:
        result = processor.process_dpr_document(f'{path}/{filename}')
        
        print(f"Processing completed in {result.processing_time:.2f} seconds")
        print(f"Extracted {result.processing_metadata['requirements_extracted']} requirements")
        print(f"Quality score: {result.quality_metrics['overall_score']:.1f}/10")
        
        # Save results
        with open(f"{filename}_requirements_only_2.md", "w", encoding='utf-8') as f:
            f.write(result.requirements_document)
            
        print(f"Requirements document saved to '{filename}_requirements.md'")

        
        
    except Exception as e:
        print(f"Error processing document: {e}")


if __name__ == "__main__":
    os.environ["GEMINI_API_KEY"] = "AIzaSyDZlba6owD92R-PFb0CgkEpHUEdcehOTRg"

    # Set up environment variables
    if not os.getenv("GEMINI_API_KEY"):
        print("Please set GEMINI_API_KEY environment variable")
        exit(1)
    
    # Run example
    example_usage()