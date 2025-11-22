from google import genai

CLIENT = genai.Client(
    api_key="gemini_api_key"
)

def refine(filename: str):
    pdf_file = CLIENT.files.upload(file = filename, config={"mime_type": "application/pdf"})

    response = CLIENT.models.generate_content(
        model = 'gemini-2.5-flash',
        contents = [
        pdf_file,
        "\n\n",
        """
        You are a civil engineering cost estimation expert.

        TASK: Extract a comprehensive Bill of Quantities (BOQ) from the provided DPR document.

        REQUIREMENTS:
        1. Identify all major construction items
        2. Focus on: Earthwork, Pavement layers, Structures, Drainage, Road furniture
        3. Get me current market rates for materials and labor in INR and if you can't find the rate, use standard rates.
        4. Include item number, description, unit, quantity, rate, and amount

        RETURN FORMAT EXAMPLE (JSON):
        [
            {
                "item_no": "1.0",
                "description": "Full description of work item",
                "unit": "Cum/Sqm/Rmt/LS/Nos",
                "quantity": numeric_value,
                "rate_inr": numeric_value,
                "amount_inr": numeric_value
            }
        ]

        Extract at least 20 major items.
        """
        ],
    )
    return response.text

if __name__ == "__main__":
    filename = 'DPR-Pkg-X-021220.pdf'
    pdf_file = f"{filename}_refined_requirements_only_2.pdf"

    print(refine(pdf_file))




