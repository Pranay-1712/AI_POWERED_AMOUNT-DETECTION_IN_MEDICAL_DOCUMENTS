# Modified app/classification.py for Gemini API

import google.generativeai as genai
import json
import re
import logging
from typing import List, Dict, Tuple
from .models import ClassificationOutput, ClassifiedAmount
from .prompts import get_classification_prompt

logger = logging.getLogger(__name__)

class GeminiClassificationService:
    """Service to classify amounts using Google Gemini API"""
    
    def __init__(self, gemini_api_key: str):
        """Initialize classification service with Gemini API"""
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Classification categories as per problem statement
        self.amount_types = [
            'total_bill', 'paid', 'due', 'discount', 'tax',
            'consultation', 'medicine', 'test', 'other'
        ]
        
        # Keywords for rule-based fallback classification
        self.keyword_mapping = {
            'total_bill': ['total', 'bill', 'amount', 'sum', 'grand total', 'net amount'],
            'paid': ['paid', 'payment', 'received', 'cash', 'advance'],
            'due': ['due', 'balance', 'pending', 'outstanding', 'remaining'],
            'discount': ['discount', 'off', 'reduction', 'concession', 'rebate'],
            'tax': ['tax', 'gst', 'vat', 'service tax', 'cgst', 'sgst', 'igst'],
            'consultation': ['consultation', 'doctor', 'visit', 'checkup', 'fee'],
            'medicine': ['medicine', 'drug', 'tablet', 'capsule', 'prescription', 'pharmacy'],
            'test': ['test', 'lab', 'report', 'scan', 'x-ray', 'blood', 'urine', 'pathology']
        }

    def extract_context_windows(self, text: str, amounts: List[float], window_size: int = 50) -> Dict[float, str]:
        """Extract surrounding context for each amount"""
        contexts = {}
        
        for amount in amounts:
            # Convert amount to possible string representations
            amount_strings = [
                str(int(amount)),  # 1200
                str(amount),       # 1200.0
                f"{amount:.2f}",   # 1200.00
                f"{int(amount):,}" if amount >= 1000 else str(int(amount))  # 1,200
            ]
            
            best_context = ""
            best_match_length = 0
            
            for amount_str in amount_strings:
                # Find all occurrences of this amount string
                for match in re.finditer(re.escape(amount_str), text):
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # Extract context window
                    context_start = max(0, start_pos - window_size)
                    context_end = min(len(text), end_pos + window_size)
                    context = text[context_start:context_end].strip()
                    
                    # Choose the context with more meaningful content
                    if len(context) > best_match_length:
                        best_context = context
                        best_match_length = len(context)
            
            contexts[amount] = best_context if best_context else f"Amount: {amount}"
        
        return contexts

    def create_gemini_prompt(self, text: str, amounts: List[float], contexts: Dict[float, str]) -> str:
        """Create a well-structured prompt for Gemini"""
        return get_classification_prompt(text)

    def classify_with_gemini(self, text: str) -> List[ClassifiedAmount]:
        """Use Gemini to classify amounts based on context"""
        
        try:
            # Create prompt
            prompt = self.create_gemini_prompt(text, [], {})
            
            logger.info(f"Sending classification request to Gemini for text analysis")
            
            # Call Gemini API
            response = self.model.generate_content(prompt)
            
            # Parse response
            response_content = response.text.strip()
            
            # Clean response (remove any non-JSON content)
            json_start = response_content.find('{')
            json_end = response_content.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_content = response_content[json_start:json_end]
                classification_result = json.loads(json_content)
                
                # Extract amounts array from the response
                amounts_array = classification_result.get('amounts', [])
                
                # Convert to ClassifiedAmount objects
                classified_amounts = []
                for item in amounts_array:
                    if isinstance(item, dict) and 'value' in item and 'type' in item:
                        classified_amounts.append(ClassifiedAmount(
                            type=item['type'],
                            value=float(item['value']),
                            context=item.get('source', item.get('context', 'Gemini classification')),
                            name=item.get('name', None)
                        ))
                
                logger.info(f"Gemini successfully classified {len(classified_amounts)} amounts")
                return classified_amounts
            
            else:
                logger.warning("Gemini response does not contain valid JSON")
                raise ValueError("Invalid Gemini response format")
                
        except Exception as e:
            logger.error(f"Gemini classification failed: {e}")
            raise

    def fallback_rule_based_classification(self, text: str, amounts: List[float]) -> List[ClassifiedAmount]:
        """Rule-based fallback classification when Gemini fails"""
        
        logger.info("Using rule-based fallback classification")
        
        classified_amounts = []
        text_lower = text.lower()
        
        # Extract context for each amount
        contexts = self.extract_context_windows(text, amounts, window_size=30)
        
        for amount in amounts:
            context = contexts.get(amount, "").lower()
            combined_text = (text_lower + " " + context).lower()
            
            # Classify based on keyword matching
            classified_type = "other"  # Default
            confidence_score = 0.0
            reasoning = "rule-based classification"
            
            # Check each category
            for category, keywords in self.keyword_mapping.items():
                category_score = 0
                matched_keywords = []
                
                for keyword in keywords:
                    if keyword in combined_text:
                        # Give higher weight to keywords found near the amount
                        if keyword in context:
                            category_score += 2
                            matched_keywords.append(keyword)
                        else:
                            category_score += 1
                            matched_keywords.append(keyword)
                
                # Select category with highest score
                if category_score > confidence_score:
                    confidence_score = category_score
                    classified_type = category
                    reasoning = f"matched keywords: {', '.join(matched_keywords)}"
            
            # Special logic for amount patterns
            if classified_type == "other":
                classified_type, reasoning = self._apply_amount_pattern_rules(amount, amounts, text_lower)
            
            classified_amounts.append(ClassifiedAmount(
                type=classified_type,
                value=amount,
                context=reasoning,
                name=None  # Fallback doesn't extract specific names
            ))
        
        return classified_amounts

    def _apply_amount_pattern_rules(self, amount: float, all_amounts: List[float], text: str) -> Tuple[str, str]:
        """Apply pattern-based rules for classification"""
        
        # If it's the largest amount, likely total
        if amount == max(all_amounts) and len(all_amounts) > 1:
            return "total_bill", "largest amount in document"
        
        # If it's a round number and largest, likely total
        if amount == int(amount) and amount >= 1000:
            return "total_bill", "large round number"
        
        # If it's a small amount, likely consultation or medicine
        if amount <= 500:
            return "consultation", "small amount pattern"
        
        # If multiple amounts and this is second largest, might be paid amount
        if len(all_amounts) > 1:
            sorted_amounts = sorted(all_amounts, reverse=True)
            if amount == sorted_amounts[1]:
                return "paid", "second largest amount"
        
        return "other", "no clear pattern identified"

    def calculate_classification_confidence(self, 
                                          classified_amounts: List[ClassifiedAmount],
                                          used_gemini: bool) -> float:
        """Calculate confidence score for classification step"""
        
        if not classified_amounts:
            return 0.0
        
        # Base confidence depends on method used
        base_confidence = 0.8 if used_gemini else 0.6
        
        # Boost confidence if we have variety in classifications
        unique_types = len(set(item.type for item in classified_amounts))
        if unique_types > 1:
            base_confidence += 0.1
        
        # Boost confidence if we have common medical bill patterns
        types_found = [item.type for item in classified_amounts]
        
        # Good patterns: total + paid, total + due, consultation + medicine
        good_patterns = [
            ['total_bill', 'paid'],
            ['total_bill', 'due'],
            ['consultation', 'medicine'],
            ['total_bill', 'tax']
        ]
        
        for pattern in good_patterns:
            if all(t in types_found for t in pattern):
                base_confidence += 0.05
        
        # Penalty if too many "other" classifications
        other_count = sum(1 for item in classified_amounts if item.type == "other")
        if other_count > len(classified_amounts) * 0.5:  # More than 50% are "other"
            base_confidence -= 0.2
        
        return min(base_confidence, 1.0)

    def process_amounts(self, text: str, amounts: List[float]) -> ClassificationOutput:
        """
        Main processing method for Step 3: Classification by Context
        """
        try:
            logger.info(f"Starting classification of text with {len(amounts)} amounts")
            
            classified_amounts = []
            used_gemini = False
            
            try:
                # Try Gemini classification first (now analyzes entire text)
                classified_amounts = self.classify_with_gemini(text)
                used_gemini = True
                
            except Exception as e:
                logger.warning(f"Gemini classification failed: {e}, using rule-based fallback")
                classified_amounts = self.fallback_rule_based_classification(text, amounts)
                used_gemini = False
            
            # Calculate confidence
            confidence = self.calculate_classification_confidence(classified_amounts, used_gemini)
            
            logger.info(f"Classification completed: {len(classified_amounts)} amounts classified with confidence {confidence:.2f}")
            
            # Create output following exact schema
            result = ClassificationOutput(
                amounts=classified_amounts,
                confidence=round(confidence, 2)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Classification processing error: {e}")
            raise ValueError(f"Classification failed: {str(e)}")

# Note: This service will be instantiated in main.py with the Gemini API key