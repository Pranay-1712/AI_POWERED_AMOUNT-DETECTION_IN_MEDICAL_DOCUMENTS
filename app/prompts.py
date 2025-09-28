# app/prompts.py
"""
Prompt templates for AI services
Centralized prompt management for the Medical Amount Detection API
"""

def get_classification_prompt(text: str) -> str:
    """
    Generate the classification prompt for Gemini AI
    
    Args:
        text: The input text to classify
        
    Returns:
        Formatted prompt string for Gemini
    """
    
    prompt = f"""You are an AI assistant specialized in analyzing medical bills and receipts. Your task is to classify financial amounts based on their context.

Input (text):
{text}

Step 3 - Classification by Context
Use surrounding text to label amounts and identify specific items/services.

Expected Output (JSON):
{{
 "amounts": [
  {{"type":"total_bill","value":1200,"name":"Total Amount","source":"text: 'Total: INR 1200'"}},
  {{"type":"paid","value":1000,"name":"Advance Payment","source":"text: 'Paid: 1000'"}},
  {{"type":"due","value":200,"name":"Balance Due","source":"text: 'Due: 200'"}},
  {{"type":"medicine","value":150,"name":"Paracetamol 200mg","source":"text: 'Paracetamol 200mg: Rs 150'"}},
  {{"type":"consultation","value":500,"name":"Doctor Consultation","source":"text: 'Doctor Fee: Rs 500'"}}
 ],
 "confidence": 0.80
}}

CRITICAL: The "source" field must contain the CORRECTED text, not the original OCR text with errors.
Example: If OCR shows "Cone uttation: Rs 200", the source should be "text: 'Consultation: Rs 200'" (corrected), NOT "text: 'Cone uttation: Rs 200'" (original with errors).

INSTRUCTIONS:
1. FIRST: Correct OCR errors in the text by fixing spelling mistakes, character misreads, and word distortions based on medical context
2. THEN: Analyze each amount in the corrected text and determine its type based on surrounding context
3. Identify specific items, services, or descriptions (e.g., medicine names, test names, service descriptions)
4. Assign appropriate types based on context (total_bill, paid, due, discount, tax, consultation, medicine, test, room_charges, etc.)
5. Extract specific names/details when available (e.g., "Paracetamol 200mg", "Blood Test", "Room Charges", "Doctor Fee")
6. Include the source text context for each amount using the CORRECTED text (e.g., "text: 'Total: INR 1200'") - NEVER use the original OCR text with errors. Always show the corrected version in the source field.
7. Calculate confidence based on how clear the context is for each amount
8. Return only the JSON object with amounts array and confidence score

Return only the JSON object, no additional text:"""
    
    return prompt
