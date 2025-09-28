# app/normalization.py
"""
Step 2: Normalization Service
Fix OCR digit errors and map to numbers with high accuracy
"""
import re
import logging
from typing import List, Dict, Tuple, Optional
from .models import NormalizationOutput

logger = logging.getLogger(__name__)

class NormalizationService:
    """Service to normalize and clean OCR-extracted numeric tokens"""
    
    def __init__(self):
        """Initialize normalization service with error correction mappings"""
        
        # Common OCR digit errors and their corrections
        self.ocr_digit_corrections = {
            'O': '0',  # Letter O to zero
            'o': '0',  # Lowercase o to zero
            'I': '1',  # Letter I to one
            'l': '1',  # Lowercase L to one
            '|': '1',  # Pipe to one
            'S': '5',  # Letter S to five
            's': '5',  # Lowercase s to five
            'G': '6',  # Letter G to six
            'B': '8',  # Letter B to eight
            'g': '9',  # Lowercase g to nine
            'Z': '2',  # Letter Z to two
        }
        
        # Character patterns that should be removed/replaced
        self.cleanup_patterns = [
            (r'[,\s]+', ''),  # Remove commas and spaces within numbers
            (r'\.{2,}', '.'),  # Multiple dots to single dot
            (r'^\.+|\.+$', ''),  # Leading/trailing dots
        ]
        
        # Validation ranges for different amount types
        self.amount_ranges = {
            'consultation': (50, 5000),
            'medicine': (10, 10000),
            'test': (100, 15000),
            'total_bill': (100, 100000),
            'general': (0.01, 100000)
        }

    def apply_ocr_corrections(self, token: str) -> str:
        """Apply OCR digit error corrections"""
        corrected = token
        
        # Apply character-by-character corrections
        for wrong_char, correct_char in self.ocr_digit_corrections.items():
            corrected = corrected.replace(wrong_char, correct_char)
        
        return corrected

    def clean_token(self, token: str) -> str:
        """Clean and standardize a numeric token"""
        if not token:
            return ""
        
        # Start with OCR corrections
        cleaned = self.apply_ocr_corrections(token)
        
        # Apply cleanup patterns
        for pattern, replacement in self.cleanup_patterns:
            cleaned = re.sub(pattern, replacement, cleaned)
        
        # Handle special cases
        cleaned = self._handle_special_cases(cleaned)
        
        return cleaned

    def _handle_special_cases(self, token: str) -> str:
        """Handle special formatting cases"""
        
        # Handle Indian number format (lakhs/crores)
        # 1,23,456 -> 123456
        if ',' in token:
            # Validate Indian comma format
            parts = token.split(',')
            if len(parts) >= 2:
                # First part: 1-3 digits, middle parts: exactly 2 digits, last part: 2-3 digits
                if (len(parts[0]) <= 3 and parts[0].isdigit() and
                    all(len(part) == 2 and part.isdigit() for part in parts[1:-1]) and
                    1 <= len(parts[-1]) <= 3 and parts[-1].replace('.', '').isdigit()):
                    token = token.replace(',', '')
        
        # Handle percentage signs
        if token.endswith('%'):
            token = token[:-1]
        
        # Handle currency symbols at the beginning
        token = re.sub(r'^[₹$€£]+', '', token)
        
        # Ensure only one decimal point
        decimal_count = token.count('.')
        if decimal_count > 1:
            # Keep only the last decimal point
            parts = token.split('.')
            token = '.'.join(parts[:-1]).replace('.', '') + '.' + parts[-1]
        
        return token

    def validate_amount(self, amount: float, amount_type: str = 'general') -> bool:
        """Validate if the amount is within reasonable ranges"""
        
        min_val, max_val = self.amount_ranges.get(amount_type, self.amount_ranges['general'])
        
        return min_val <= amount <= max_val

    def convert_to_number(self, token: str) -> Optional[float]:
        """Convert cleaned token to a float number"""
        if not token:
            return None
        
        try:
            # Handle empty or invalid tokens
            if not re.match(r'^\d*\.?\d+$', token):
                return None
            
            # Convert to float
            number = float(token)
            
            # Basic sanity check
            if number < 0:
                return None
            
            # Round to 2 decimal places for currency
            return round(number, 2)
            
        except (ValueError, TypeError):
            return None

    def calculate_normalization_confidence(self, 
                                         original_tokens: List[str], 
                                         normalized_amounts: List[float]) -> float:
        """Calculate confidence score for normalization step"""
        
        if not original_tokens:
            return 0.0
        
        # Base confidence from success rate
        success_rate = len(normalized_amounts) / len(original_tokens)
        confidence = success_rate * 0.7
        
        # Boost confidence for reasonable amounts
        reasonable_count = sum(1 for amount in normalized_amounts 
                             if self.validate_amount(amount))
        if normalized_amounts:
            reasonable_ratio = reasonable_count / len(normalized_amounts)
            confidence += reasonable_ratio * 0.2
        
        # Penalty for too many or too few amounts
        if len(normalized_amounts) > 10:  # Too many amounts detected
            confidence *= 0.8
        elif len(normalized_amounts) == 0:  # No amounts detected
            confidence = 0.0
        
        # Boost confidence if amounts are in expected ranges
        if normalized_amounts:
            avg_amount = sum(normalized_amounts) / len(normalized_amounts)
            if 100 <= avg_amount <= 5000:  # Typical medical bill range
                confidence += 0.1
        
        return min(confidence, 1.0)

    def process_tokens(self, raw_tokens: List[str]) -> NormalizationOutput:
        """
        Main processing method for Step 2: Normalization
        
        Args:
            raw_tokens: List of raw tokens from OCR step
            
        Returns:
            NormalizationOutput: Following the exact schema from problem statement
        """
        try:
            logger.info(f"Starting normalization of {len(raw_tokens)} tokens")
            
            normalized_amounts = []
            processing_log = []
            
            for i, token in enumerate(raw_tokens):
                original_token = token
                
                # Step 1: Clean the token
                cleaned_token = self.clean_token(token)
                
                # Step 2: Convert to number
                amount = self.convert_to_number(cleaned_token)
                
                if amount is not None:
                    # Step 3: Validate the amount
                    if self.validate_amount(amount):
                        normalized_amounts.append(amount)
                        processing_log.append(f"Token '{original_token}' -> {amount}")
                    else:
                        processing_log.append(f"Token '{original_token}' -> {amount} (rejected: out of range)")
                else:
                    processing_log.append(f"Token '{original_token}' -> None (conversion failed)")
            
            # Remove duplicates while preserving order
            seen = set()
            unique_amounts = []
            for amount in normalized_amounts:
                if amount not in seen:
                    unique_amounts.append(amount)
                    seen.add(amount)
            
            # Calculate confidence
            confidence = self.calculate_normalization_confidence(raw_tokens, unique_amounts)
            
            # Log processing summary
            logger.info(f"Normalization completed: {len(unique_amounts)} valid amounts from {len(raw_tokens)} tokens")
            for log_entry in processing_log:
                logger.debug(log_entry)
            
            # Create output following exact schema
            result = NormalizationOutput(
                normalized_amounts=unique_amounts,
                normalization_confidence=round(confidence, 2)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Normalization processing error: {e}")
            raise ValueError(f"Normalization failed: {str(e)}")

# Singleton instance
normalization_service = NormalizationService()