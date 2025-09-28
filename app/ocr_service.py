# app/ocr_service.py - SIMPLE, RELIABLE VERSION
"""
Step 1: OCR/Text Extraction Service
Simple, reliable approach focused on extracting amounts correctly
"""
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import re
import io
import logging
from typing import List, Tuple, Optional
from .models import OCROutput

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCRService:
    """Simple, reliable OCR service for medical documents"""
    
    def __init__(self):
        """Initialize OCR service with simple configuration"""
        # Currency patterns for detection
        self.currency_patterns = {
            'INR': [r'(?:Rs\.?|INR|₹)', r'rupees?', r'paisa'],
            'USD': [r'\$', r'USD', r'dollars?'],
            'EUR': [r'€', r'EUR', r'euros?']
        }

    def extract_text_from_image(self, image_data: bytes) -> Tuple[str, float]:
        """Simple, reliable text extraction"""
        try:
            # Open image with PIL
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Simple resize if image is too small
            width, height = image.size
            if width < 1500 or height < 1500:
                scale = max(1500/width, 1500/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Try multiple simple OCR approaches
            ocr_configs = [
                '--oem 3 --psm 6',  # Uniform block
                '--oem 3 --psm 4',  # Single column
                '--oem 3 --psm 3',  # Auto segmentation
            ]
            
            best_text = ""
            best_confidence = 0
            
            for config in ocr_configs:
                try:
                    # Extract text
                    text = pytesseract.image_to_string(image, config=config)
                    
                    # Get confidence data
                    data = pytesseract.image_to_data(
                        image, config=config, output_type=pytesseract.Output.DICT
                    )
                    
                    # Calculate confidence
                    confidences = [int(c) for c in data['conf'] if int(c) > 0]
                    avg_conf = sum(confidences) / len(confidences) if confidences else 0
                    
                    # Score based on amount detection
                    amount_count = len(re.findall(r'\d{3,}', text))
                    score = avg_conf + (amount_count * 10)
                    
                    if score > best_confidence:
                        best_confidence = score
                        best_text = text
                        
                except Exception as e:
                    logger.warning(f"OCR config failed: {e}")
                    continue
            
            # Normalize confidence
            final_confidence = min(best_confidence / 100.0, 1.0)
            
            logger.info(f"Simple OCR extracted text with confidence {final_confidence:.2f}")
            logger.info(f"Raw OCR text: {best_text}")
            
            return best_text, final_confidence
            
        except Exception as e:
            logger.error(f"Simple OCR failed: {e}")
            raise ValueError(f"OCR processing failed: {e}")
    

    def extract_numeric_tokens(self, text: str) -> List[str]:
        """Extract amounts using comprehensive patterns"""
        
        logger.info(f"Extracting from text: {text}")
        
        all_tokens = []
        
        # Pattern 1: Amounts with commas and decimals (4,000.00, 2,765.54, 15,143.54)
        pattern1 = r'(?<!-)\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b'
        matches1 = re.findall(pattern1, text)
        all_tokens.extend(matches1)
        logger.info(f"Pattern 1 (x,xxx.xx): {matches1}")
        
        
        # Pattern 3: Round amounts (5000, but exclude years and IDs)
        pattern3 = r'\b(\d{4,5})(?!\d)\b'
        matches3 = re.findall(pattern3, text)
        # Filter out years and obvious IDs
        filtered3 = [m for m in matches3 if not (2020 <= int(m) <= 2030) and int(m) >= 1000]
        all_tokens.extend(filtered3)
        logger.info(f"Pattern 3 (round amounts): {filtered3}")
        
        # Pattern 4: Currency prefixed amounts (Rs.5000, ₹1000)
        pattern4 = r'(?:Rs\.?|₹)\s*(\d{3,6})'
        matches4 = re.findall(pattern4, text, re.IGNORECASE)
        all_tokens.extend(matches4)
        logger.info(f"Pattern 4 (currency prefix): {matches4}")
        
        # Pattern 5: Contextual amounts (Amount: 15143.54)
        pattern5 = r'(?:Amount|Total|Bill|Paid|Due)\s*:?\s*(\d{1,6}(?:\.\d{2})?)'
        matches5 = re.findall(pattern5, text, re.IGNORECASE)
        all_tokens.extend(matches5)
        logger.info(f"Pattern 5 (contextual): {matches5}")
        
        # Clean and validate all tokens
        valid_tokens = []
        seen = set()
        
        for token in all_tokens:
            cleaned = self._clean_token(token)
            if cleaned and cleaned not in seen and self._is_valid_amount(cleaned):
                valid_tokens.append(cleaned)
                seen.add(cleaned)
        
        logger.info(f"Final valid tokens: {valid_tokens}")
        return valid_tokens

    def _clean_token(self, token: str) -> Optional[str]:
        """Clean token and preserve format"""
        if not token:
            return None
        
        # Remove any non-numeric characters except comma and dot
        cleaned = re.sub(r'[^\d.,]', '', token.strip())
        
        # Remove leading/trailing punctuation
        cleaned = cleaned.strip('.,')
        
        if not cleaned:
            return None
        
        # For display, keep original format if it has commas
        # For calculation, we'll handle comma removal elsewhere
        return cleaned

    def _is_valid_amount(self, token: str) -> bool:
        """Check if token is a valid medical bill amount"""
        try:
            # Remove commas for calculation
            amount = float(token.replace(',', ''))
            
            # Medical bill range: Rs 40 to Rs 5,00,000
            return 40 <= amount <= 500000
            
        except ValueError:
            return False

    def detect_currency(self, text: str) -> str:
        """Detect currency from text"""
        text_lower = text.lower()
        
        for currency, patterns in self.currency_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return currency
        
        return "INR"

    def calculate_extraction_confidence(self, text: str, tokens: List[str], ocr_confidence: float) -> float:
        """Calculate confidence based on results"""
        if not tokens:
            return 0.0
        
        # Base OCR confidence
        confidence = ocr_confidence * 0.6
        
        # Boost for number of tokens found
        confidence += min(len(tokens) * 0.1, 0.3)
        
        # Boost for medical keywords
        medical_words = ['bill', 'amount', 'total', 'room', 'pharmacy', 'hospital']
        keyword_count = sum(1 for word in medical_words if word in text.lower())
        confidence += min(keyword_count * 0.02, 0.1)
        
        return min(confidence, 1.0)

    def process_input(self, image_data: bytes = None, text: str = None) -> OCROutput:
        """Main processing method"""
        try:
            if image_data:
                extracted_text, ocr_confidence = self.extract_text_from_image(image_data)
            elif text:
                extracted_text = text
                ocr_confidence = 1.0
            else:
                raise ValueError("Either image_data or text must be provided")
            
            # Extract tokens
            tokens = self.extract_numeric_tokens(extracted_text)
            
            # Detect currency
            currency_hint = self.detect_currency(extracted_text)
            
            # Calculate confidence
            final_confidence = self.calculate_extraction_confidence(
                extracted_text, tokens, ocr_confidence
            )
            
            result = OCROutput(
                raw_tokens=tokens,
                currency_hint=currency_hint,
                confidence=round(final_confidence, 2)
            )
            
            logger.info(f"OCR completed: {len(tokens)} tokens, confidence: {final_confidence:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"OCR processing error: {e}")
            raise ValueError(f"OCR processing failed: {str(e)}")

# Singleton instance
ocr_service = OCRService()