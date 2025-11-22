import os
from typing import List, Dict
from datetime import datetime
from langchain.schema import Document
from autogen import ConversableAgent

from vector_store import RAGVectorStore
from google import genai

CLIENT = genai.Client(
    api_key="AIzaSyDZlba6owD92R-PFb0CgkEpHUEdcehOTRg"
)

class TechnicalSpecialistAgent(ConversableAgent):
    """Technical requirements specialist with RAG context"""
    
    def __init__(self, vector_store: RAGVectorStore):
        super().__init__(
            name="TechnicalSpecialist",
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
            You are a technical architecture specialist focusing on:
            - Hardware and software requirements
            - System integration specifications
            - Performance and scalability requirements
            - Security and compliance specifications
            - Infrastructure and deployment needs
            
            Your role:
            - Enhance requirements with technical details
            - Add measurable performance criteria
            - Identify technical constraints and dependencies  
            - Specify technology standards and protocols
            - Flag potential technical risks and challenges
            """
        )
        self.vector_store = vector_store
    
    def enhance_technical_requirements(self, requirements: List[Dict]) -> List[Dict]:
        """Enhance requirements with technical specifications"""
        enhanced_requirements = []
        
        for req in requirements:
            if req.get('type') == 'technical' or self.needs_technical_enhancement(req):
                # Get technical context
                tech_query = f"technical specifications {req.get('description', '')}"
                # tech_context = self.vector_store.retrieve_relevant_chunks(tech_query, k=3)
                
                # # Enhance the requirement
                # enhanced_req = self.enhance_single_requirement(req, tech_context)
                # enhanced_requirements.append(enhanced_req)

                enhanced_requirements.append(req)
            else:
                enhanced_requirements.append(req)
        
        return enhanced_requirements
    
    def needs_technical_enhancement(self, req: Dict) -> bool:
        """Determine if requirement needs technical enhancement"""
        description = req.get('description', '').lower()
        tech_keywords = [
            'system', 'software', 'hardware', 'database', 'network',
            'performance', 'security', 'integration', 'api', 'server'
        ]
        
        return any(keyword in description for keyword in tech_keywords)
    
    def enhance_single_requirement(self, req: Dict, context_chunks: List[Document]) -> Dict:
        """Enhance a single requirement with technical details"""
        context_text = "\n".join([chunk.page_content for chunk in context_chunks[:2]])
        
        prompt = f"""
        Enhance this requirement with technical specifications:
        
        ORIGINAL REQUIREMENT:
        ID: {req.get('id', '')}
        Description: {req.get('description', '')}
        Type: {req.get('type', '')}
        
        TECHNICAL CONTEXT:
        {context_text}
        
        ENHANCEMENT GUIDELINES:
        1. Add specific technical criteria (response times, throughput, capacity)
        2. Specify technology constraints and standards
        3. Define integration requirements and APIs
        4. Add security and compliance considerations
        5. Include scalability and performance metrics
        6. Identify technical dependencies
        
        Provide enhanced requirement maintaining original ID and adding technical details.
        Focus on measurable, testable specifications.
        """
        
        try:
            response = CLIENT.models.generate_content(
                model= 'gemini-2.5-flash',
                contents = prompt
            )
            enhanced_req = self.parse_technical_enhancement(response.text, req)
            return enhanced_req
        except Exception as e:
            print(f"Error enhancing requirement {req.get('id')}: {e}")
            return req
    
    def parse_technical_enhancement(self, response: str, original_req: Dict) -> Dict:
        """Parse technical enhancement response"""
        enhanced_req = original_req.copy()
        
        # Extract technical specifications from response
        lines = response.split('\n')
        
        technical_specs = []
        performance_criteria = []
        dependencies = []
        
        for line in lines:
            line = line.strip()
            if 'performance:' in line.lower():
                performance_criteria.append(line.split(':', 1)[1].strip())
            elif 'dependency:' in line.lower():
                dependencies.append(line.split(':', 1)[1].strip())
            elif any(keyword in line.lower() for keyword in ['specification:', 'technical:', 'constraint:']):
                technical_specs.append(line.split(':', 1)[1].strip())
        
        # Update requirement with enhancements
        if technical_specs:
            enhanced_req['technical_specifications'] = technical_specs
        if performance_criteria:
            enhanced_req['performance_criteria'] = performance_criteria
        if dependencies:
            enhanced_req['technical_dependencies'] = dependencies
        
        # Mark as technically enhanced
        enhanced_req['enhanced'] = True
        enhanced_req['enhancement_timestamp'] = datetime.now().isoformat()
        
        return enhanced_req