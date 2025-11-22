from google import genai
from typing import List, Dict
import re
from collections import defaultdict
from spire.doc import Document, FileFormat
from markdown_pdf import MarkdownPdf, Section
from main import RAGDPRProcessor
from vector_store import RAGVectorStore

CLIENT = genai.Client(
    api_key="gemini_api_key"
)

def extract_requirements_from_md(md_path):
    requirements_dict = defaultdict(list)
    current_section = None

    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # Detect section headers (### Section Name)
        if line.startswith("### "):
            current_section = line.replace("###", "").strip()
            continue

        # Detect bullet points (* item or - item)
        if line.startswith("*") or line.startswith("-"):
            if current_section:
                # Remove leading * / - and extra spaces
                clean_line = re.sub(r"^[\*\-]\s*", "", line)
                requirements_dict[current_section].append(clean_line)

    return dict(requirements_dict)

class RefineRequirements:
    def __init__(self, filename: str, source_pdf_filename: str):
        self.client = CLIENT
        self.filename = filename
        self.source_pdf_filename = source_pdf_filename

    def get_chunks_for_complex_requirements(self, requirements: List) -> List:
        embedding_model = "BAAI/bge-base-en-v1.5"

        processor = RAGDPRProcessor()
        document_chunks = processor.load_and_chunk_document(self.source_pdf_filename)
        vector_store = RAGVectorStore(embedding_model_name=embedding_model)

        vector_store.create_vector_store(document_chunks)
        results = []

        for i in requirements:
            chunks = [chunk.content for (chunk, _) in vector_store.retrieve_relevant_chunks(i)]
            for j in chunks:
                if j not in results:
                    results.append(j)

        return results


    def refine(self):
        file = self.client.files.upload(file = self.filename, config={"mime_type": "text/markdown"})

        response = self.client.models.generate_content(
            model = 'gemini-2.5-flash',
            contents = [
            file,
            "\n\n",
            f"""
            I want you to extract all the requirements as a list

            also make sure to merge similar requirements but don't change the meaning though

            also if possible split the requirements into different categories

            also consider all requirements in the file and don't miss any requirement
            also make sure to remove duplicates
            Do not add explanations, introduction text, or any additional information. Just provide the refined list of requirements.
            """
            ],
        )
        return response.text
    
if __name__ == "__main__":
    filename = 'DPR-Pkg-X-021220.pdf'

    requirements = RefineRequirements(f"{filename}_requirements_only_2.md", f"DPR/{filename}")
    #refined_requirements = requirements.refine()
    #with open(f"{filename}_refined_requirements_only_2.md", "w", encoding='utf-8') as f:
     #   f.write(refined_requirements)

    with open(f"{filename}_refined_requirements_only_2.md", "r", encoding='utf-8') as f:
       md = f.read()

    # document = Document()

    # document.LoadFromFile(f"{filename}_refined_requirements_only_2.md")
    # document.SaveToFile("ToPdf.pdf", FileFormat.PDF)
    # document.Dispose()

    pdf = MarkdownPdf(toc_level=0)
    pdf.add_section(Section(md))
    pdf.save(f"{filename}_refined_requirements_only_2.pdf")

    extracted_requirements = extract_requirements_from_md(f"{filename}_refined_requirements_only_2.md")
    result = requirements.get_chunks_for_complex_requirements(extracted_requirements['Design & Engineering - Pavement'])

    print(result)
    


