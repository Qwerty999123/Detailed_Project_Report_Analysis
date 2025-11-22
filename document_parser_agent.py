import os
from typing import List, Dict
import json
import re
from autogen import ConversableAgent

from chunks import DocumentChunk

from google import genai

CLIENT = genai.Client(
    api_key="AIzaSyDZlba6owD92R-PFb0CgkEpHUEdcehOTRg"
)

class DocumentParserAgent(ConversableAgent):
    """Document parsing and structure analysis agent"""
    
    def __init__(self):
        super().__init__(
            name="DocumentParser",
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
            You are a specialized document parsing agent focused on:
            - Analyzing document structure and organization
            - Extracting metadata (project details, stakeholders, timelines)
            - Identifying key sections and their relationships
            - Recognizing document type and domain
            - Creating structural roadmap for requirement extraction
            
            Your analysis helps other agents understand document context and focus
            their extraction efforts on relevant sections.
            """
        )
    
    def analyze_document_structure(self, chunks: List[DocumentChunk]) -> Dict:
        """Analyze overall document structure from chunks"""
        structure_analysis = {
            'document_metadata': {},
            'section_map': {},
            'content_distribution': {},
            'key_entities': [],
            'document_type': 'unknown',
            'project_domain': 'unknown',
            'total_chunks': len(chunks)
        }
        
        # Analyze chunks to understand structure
        section_counts = {}
        content_types = {}
        all_text = ""
        
        for chunk in chunks:
            section = chunk.metadata.get('section', 'unknown')
            content_type = chunk.metadata.get('content_type', 'text')
            
            section_counts[section] = section_counts.get(section, 0) + 1
            content_types[content_type] = content_types.get(content_type, 0) + 1
            
            # Collect text for analysis (limit to avoid token overflow)
            if len(all_text) < 10000:
                all_text += chunk.content + " "
        
        # Extract metadata from text analysis
        structure_analysis['document_metadata'] = self.extract_metadata_from_text(all_text)
        structure_analysis['section_map'] = section_counts
        structure_analysis['content_distribution'] = content_types
        structure_analysis['key_entities'] = self.extract_key_entities(all_text)
        structure_analysis['document_type'] = self.classify_document_type(all_text)
        structure_analysis['project_domain'] = self.classify_project_domain(all_text)
        
        return structure_analysis
    
    def extract_metadata_from_text(self, text: str) -> Dict:
        """Extract document metadata using LLM"""
        prompt = f"""
        Extract key metadata from this document text:
        
        {text[:5000]}  # Limit text to avoid token overflow
        
        Extract:
        1. Project/System name
        2. Organization/Department
        3. Date/Timeline information
        4. Key stakeholders mentioned
        5. Budget/Cost information
        6. Geographic scope
        
        Provide structured JSON output with extracted metadata.
        """
        
        try:
            response = CLIENT.models.generate_content(
                model= 'gemini-2.5-flash',
                contents = prompt
            )
            if type(response.text) == str:
                return self.parse_metadata_response(response.text)
            else:
                print("None type in \"extract metadata from text\" function")
        except Exception as e:
            print(f"Error extracting metadata: {e}")
            return {}
    
    def parse_metadata_response(self, response: str) -> Dict:
        """Parse metadata extraction response"""
        try:
            if "{" in response and "}" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
                return json.loads(json_str)
        except:
            pass
        
        # Fallback parsing
        metadata = {}
        lines = response.split('\n')
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                if value and not value.lower() in ['unknown', 'not found', 'n/a']:
                    metadata[key] = value
        
        return metadata
    
    def extract_key_entities(self, text: str) -> List[str]:
        """Extract key entities from text"""
        # Simple entity extraction using patterns
        entities = []
        
        # Look for common project entities
        entity_patterns = [
            r'\b[A-Z][A-Za-z]*\s+(?:System|Application|Platform|Project)\b',
            r'\b(?:Ministry|Department|Division)\s+of\s+[A-Za-z\s]+\b',
            r'\b[A-Z]{2,10}\b',  # Acronyms
            r'\b(?:Rs\.?\s*)?[\d,]+(?:\.\d+)?\s*(?:crore|lakh|million|billion)\b'  # Financial amounts
        ]
        
        for pattern in entity_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.extend(matches)
        
        # Remove duplicates and limit
        return list(set(entities))[:20]
    
    def classify_document_type(self, text: str) -> str:
        """Classify document type"""
        text_lower = text.lower()
        
        if 'detailed project report' in text_lower or 'dpr' in text_lower:
            return 'dpr'
        elif 'requirement' in text_lower and 'specification' in text_lower:
            return 'requirements_specification'
        elif 'system design' in text_lower:
            return 'system_design'
        elif 'implementation plan' in text_lower:
            return 'implementation_plan'
        else:
            return 'project_document'
    
    def classify_project_domain(self, text: str) -> str:
        """Classify project domain"""
        text_lower = text.lower()
        
        domain_keywords = {
            'government': ['government', 'ministry', 'department', 'public', 'citizen'],
            'healthcare': ['health', 'medical', 'hospital', 'patient', 'healthcare'],
            'education': ['education', 'school', 'university', 'student', 'learning'],
            'finance': ['bank', 'financial', 'payment', 'transaction', 'finance'],
            'infrastructure': ['infrastructure', 'construction', 'building', 'facility'],
            'technology': ['software', 'system', 'application', 'digital', 'technology']
        }
        
        for domain, keywords in domain_keywords.items():
            if sum(1 for keyword in keywords if keyword in text_lower) >= 2:
                return domain
        
        return 'general'