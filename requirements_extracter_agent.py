import os
from typing import List, Dict, Tuple
import json
from langchain.schema import Document
from autogen import ConversableAgent

from vector_store import RAGVectorStore
from chunks import DocumentChunk

from google import genai

CLIENT = genai.Client(
    api_key="gemini_api_key"
)


class RAGRequirementExtractorAgent(ConversableAgent):
    """RAG-enhanced requirement extraction agent"""
    
    def __init__(self, vector_store: RAGVectorStore):
        super().__init__(
            name="RAGRequirementExtractor",
            llm_config={
                "config_list": [{
                    "model": "gemini-2.5-flash",  # Updated model name
                    "api_key": os.getenv("GEMINI_API_KEY"),
                    "api_type": "google",  # Specify Google API
                    "base_url": "https://generativelanguage.googleapis.com/v1beta"  # Gemini base URL
                }],
                "temperature": 0.1
            },
            system_message="""
            You are an expert at extracting requirements from project documents using RAG.
            
            Your expertise:
            - Extract requirements from document chunks with context awareness
            - Use retrieved context to understand implicit requirements
            - Maintain coherence across document sections
            - Cross-reference requirements from different chunks
            - Identify dependencies spanning multiple sections
            
            For each requirement:
            - Assign unique ID (REQ-[SECTION]-[NUMBER])
            - Classify type (Functional/Non-Functional/Technical/Business)
            - Set priority (Must/Should/Could/Won't)
            - Provide clear, testable description
            - Note source and dependencies
            - Flag any ambiguities
            """
        )
        self.vector_store = vector_store
        self.extracted_requirements = []
    
    def extract_requirements_from_chunks(self, chunks: List[DocumentChunk], 
                                       section_name: str = "general") -> List[Dict]:
        """Extract requirements from chunks with RAG context"""
        section_requirements = []
        
        # for chunk in chunks:
        # Get additional context for this chunk
        # print(chunk)
        # context_query = f"requirements specifications {section_name} {chunk.content[:100]}"
        # context_chunks = self.vector_store.retrieve_relevant_chunks(
        #     context_query, k=5
        # )
        
        # Create extraction prompt
        prompt = self.create_extraction_prompt(chunks, section_name)
        
        # Extract requirements
        try:
            # response = self.generate_reply([{"content": prompt, "role": "user"}])
            # if type(response) == str:
            #     chunk_requirements = self.parse_requirements_response(response)
            # elif type(response) == Dict:
            #     chunk_requirements = self.parse_requirements_response(list(response.values())[0])
            # else:
            response = CLIENT.models.generate_content(
                model= 'gemini-2.5-flash',
                contents = prompt
            )
            if len(response.text) == 0 or type(response.text) == None:
                chunk_requirements = ''
                print("Received None value from a line in \"extract_requirements_from_chunks\" function")
            else:
                chunk_requirements = self.parse_requirements_response(response.text)
            # chunk_requirements = self.parse_requirements_response(response, chunk.metadata)
            section_requirements.extend(chunk_requirements)
        except Exception as e:
            print(f"Error extracting from chunk: {e}")
        
        # Merge and deduplicate requirements
        merged_requirements = self.merge_similar_requirements(section_requirements)
        self.extracted_requirements.extend(merged_requirements)
        
        return merged_requirements
    
    def create_extraction_prompt(self, context_chunks: List[DocumentChunk], 
                               section_name: str) -> str:
        """Create a comprehensive extraction prompt"""
        
        context_text = self.format_context_chunks(context_chunks)
        
        prompt = f"""
        Extract requirements from this document chunk, using additional context for clarity:
        
        EXTRACTION GUIDELINES:
        1. Identify explicit requirements (must, shall, should, will)
        2. Infer implicit requirements from context
        3. Consider technical specifications and constraints
        4. Note business rules and compliance needs
        5. Extract performance and quality requirements

        CONTEXT:
        {context_text}
        
        OUTPUT FORMAT (JSON):
        {{
            "requirements": [
                {{
                    "id": "REQ-{section_name.upper()}-001",
                    "type": "functional|non-functional|technical|business",
                    "priority": "must|should|could|wont",
                    "description": "Clear, testable requirement description",
                    "source": "chunk reference",
                    "dependencies": ["related requirement IDs"],
                    "acceptance_criteria": ["testable criteria"],
                    "notes": "additional context or ambiguities"
                }}
            ]
        }}
        
        Focus on extracting complete, actionable requirements that can be implemented and tested.
        """
        
        return prompt
    
    def format_context_chunks(self, context_chunks: List[DocumentChunk]) -> str:
        """Format context chunks for prompt inclusion"""
        if not context_chunks:
            return "No additional context available."
        
        formatted_context = []
        for i, chunk in enumerate(context_chunks[:3]):  # Limit to top 3 for token efficiency
            section = chunk.metadata.get('section', 'Unknown')
            content_preview = chunk.content[:300] + "..." if len(chunk.content) > 300 else chunk.content
            formatted_context.append(f"[Context {i+1} - {section}]: {content_preview}")
        
        return "\n".join(formatted_context)
    
    def parse_requirements_response(self, response: str) -> List[Dict]:
        """Parse LLM response into structured requirements"""
        try:
            # Try to parse as JSON
            if "{" in response and "}" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
                
                parsed = json.loads(json_str)
                requirements = parsed.get("requirements", [])
                
                # # Add source metadata
                # for req in requirements:
                #     req["source_chunk"] = {
                #         "page": chunk_metadata.get('page'),
                #         "section": chunk_metadata.get('section'),
                #         "chunk_type": chunk_metadata.get('chunk_type')
                #     }
                
                return requirements
            else:
                # Fallback parsing for non-JSON responses
                return self.fallback_parse_requirements(response)
                
        except json.JSONDecodeError:
            return self.fallback_parse_requirements(response)
    
    def fallback_parse_requirements(self, response: str) -> List[Dict]:
        """Fallback parsing for non-JSON responses"""
        requirements = []
        lines = response.split('\n')
        
        current_req = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if 'REQ-' in line and ':' in line:
                if current_req:
                    requirements.append(current_req)
                
                current_req = {
                    "id": line.split(':')[0].strip(),
                    "description": line.split(':', 1)[1].strip(),
                    "type": "functional",  # Default
                    "priority": "should",  # Default
                    "source_chunk": ''
                }
            elif line.startswith('Type:'):
                current_req["type"] = line.replace('Type:', '').strip().lower()
            elif line.startswith('Priority:'):
                current_req["priority"] = line.replace('Priority:', '').strip().lower()
        
        if current_req:
            requirements.append(current_req)
        
        return requirements
    
    def merge_similar_requirements(self, requirements: List[Dict]) -> List[Dict]:
        """Merge similar or duplicate requirements"""
        if not requirements:
            return []
        
        merged = []
        seen_descriptions = set()
        
        for req in requirements:
            description = req.get('description', '').lower().strip()
            
            # Simple similarity check
            is_similar = any(
                self.calculate_similarity(description, seen_desc) > 0.8 
                for seen_desc in seen_descriptions
            )
            
            if not is_similar:
                merged.append(req)
                seen_descriptions.add(description)
        
        return merged
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity calculation"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        

        return len(intersection) / len(union) if union else 0.0
