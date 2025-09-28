# app/utils.py
"""
Utility functions for the Medical Amount Detection API
"""
import re
import logging
from typing import List, Tuple
from .models import AmountInfo, FinalOutput, ErrorResponse

logger = logging.getLogger(__name__)

class UtilityService:
    """Utility functions for final output generation and validation"""
    
    def __init__(self):
        """Initialize utility service"""
        self.max_context_length = 100  # Maximum characters for source context
    
    def find_source_context(self, original_text: str, amount: float, context_size: int = 50) -> str:
        """
        Find the source text context for an amount with provenance
        
        Args:
            original_text: Original document text
            amount: The amount to find context for
            context_size: Number of characters before/after the amount
        
        Returns:
            Source context string for provenance
        """
        try:
            # Convert amount to possible string representations
            amount_patterns = [
                str(int(amount)),           # 1200
                str(amount),                # 1200.0
                f"{amount:.2f}",           # 1200.00
                f"{amount:.1f}",           # 1200.0
                f"{int(amount):,}",        # 1,200 (if >= 1000)
                f"Rs {int(amount)}",       # Rs 1200
                f"INR {int(amount)}",      # INR 1200
                f"₹ {int(amount)}",        # ₹ 1200
                f"Rs.{int(amount)}",       # Rs.1200
            ]
            
            best_context = f"Amount: {amount}"
            best_score = 0
            
            for pattern in amount_patterns:
                # Find all occurrences of this pattern
                for match in re.finditer(re.escape(pattern), original_text, re.IGNORECASE):
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # Extract surrounding context
                    context_start = max(0, start_pos - context_size)
                    context_end = min(len(original_text), end_pos + context_size)
                    
                    context = original_text[context_start:context_end].strip()
                    
                    # Score this context based on meaningful content
                    score = self._score_context(context, amount)
                    
                    if score > best_score:
                        best_score = score
                        best_context = context
            
            # Truncate if too long
            if len(best_context) > self.max_context_length:
                best_context = best_context[:self.max_context_length] + "..."
            
            return best_context
            
        except Exception as e:
            logger.warning(f"Could not find context for amount {amount}: {e}")
            return f"Amount: {amount}"
    
    def _score_context(self, context: str, amount: float) -> int:
        """Score context based on how informative it is"""
        score = 0
        context_lower = context.lower()
        
        # Boost score for meaningful keywords
        meaningful_keywords = [
            'total', 'bill', 'paid', 'due', 'balance', 'consultation',
            'medicine', 'test', 'discount', 'tax', 'amount', 'fee'
        ]
        
        for keyword in meaningful_keywords:
            if keyword in context_lower:
                score += 2
        
        # Boost score for currency indicators
        currency_indicators = ['rs', 'inr', '₹', '$', 'rupees']
        for indicator in currency_indicators:
            if indicator in context_lower:
                score += 1
        
        # Boost score for longer, more descriptive context
        if len(context) > 30:
            score += 1
        
        # Penalty for contexts that are just numbers
        if context.strip().replace('.', '').replace(',', '').isdigit():
            score -= 2
        
        return score
    
    def validate_final_output(self, output: FinalOutput) -> Tuple[bool, List[str]]:
        """
        Validate the final output against schema requirements
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            # Check required fields
            if not output.currency:
                errors.append("Currency field is required")
            
            if not isinstance(output.amounts, list):
                errors.append("Amounts must be a list")
            
            if not output.status:
                errors.append("Status field is required")
            
            # Validate currency
            valid_currencies = ['INR', 'USD', 'EUR']
            if output.currency not in valid_currencies:
                errors.append(f"Currency must be one of {valid_currencies}")
            
            # Validate status
            valid_statuses = ['ok', 'no_amounts_found', 'low_confidence', 'normalization_failed', 'error']
            if output.status not in valid_statuses:
                errors.append(f"Status must be one of {valid_statuses}")
            
            # Validate amounts
            for i, amount_info in enumerate(output.amounts):
                if not isinstance(amount_info.value, (int, float)):
                    errors.append(f"Amount {i}: value must be a number")
                
                if amount_info.value < 0:
                    errors.append(f"Amount {i}: value cannot be negative")
                
                if not amount_info.type:
                    errors.append(f"Amount {i}: type is required")
                
                if not amount_info.source:
                    errors.append(f"Amount {i}: source is required")
            
            # Check for duplicate amounts (might indicate processing errors)
            amount_values = [amt.value for amt in output.amounts]
            if len(amount_values) != len(set(amount_values)):
                logger.warning("Duplicate amounts detected in final output")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            return False, errors
    
    def create_error_response(self, status: str, reason: str) -> ErrorResponse:
        """Create a standardized error response"""
        
        # Ensure status is valid
        valid_statuses = ['no_amounts_found', 'low_confidence', 'normalization_failed', 'error']
        if status not in valid_statuses:
            status = 'error'
        
        return ErrorResponse(status=status, reason=reason)
    
    def generate_final_output(self, 
                            original_text: str,
                            classified_amounts: List,
                            currency: str) -> FinalOutput:
        """
        Generate final structured output with provenance (Step 4)
        
        Args:
            original_text: Original document text
            classified_amounts: List of classified amounts from step 3
            currency: Detected currency
            
        Returns:
            FinalOutput: Following the exact schema from problem statement
        """
        try:
            logger.info(f"Generating final output for {len(classified_amounts)} amounts")
            
            amounts_output = []
            
            for classified_amount in classified_amounts:
                # Use source from classification if available, otherwise find source context
                if classified_amount.context and classified_amount.context.startswith("text: '"):
                    source_context = classified_amount.context
                else:
                    # Find source context for provenance
                    source_context = self.find_source_context(
                        original_text, 
                        classified_amount.value
                    )
                
                # Create amount info with source provenance
                amount_info = AmountInfo(
                    type=classified_amount.type,
                    value=classified_amount.value,
                    source=source_context,
                    name=classified_amount.name
                )
                
                amounts_output.append(amount_info)
            
            # Create final output
            result = FinalOutput(
                currency=currency,
                amounts=amounts_output,
                status="ok"
            )
            
            # Validate the output
            is_valid, validation_errors = self.validate_final_output(result)
            
            if not is_valid:
                logger.error(f"Final output validation failed: {validation_errors}")
                # Still return the result but log the issues
                for error in validation_errors:
                    logger.error(f"Validation error: {error}")
            
            logger.info(f"Final output generated successfully with {len(amounts_output)} amounts")
            
            return result
            
        except Exception as e:
            logger.error(f"Final output generation failed: {e}")
            raise ValueError(f"Could not generate final output: {str(e)}")
    
    def sanitize_text_input(self, text: str) -> str:
        """Sanitize and clean text input"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Remove any potential harmful characters (basic sanitization)
        # Keep alphanumeric, common punctuation, and currency symbols
        cleaned = re.sub(r'[^\w\s.,;:!?₹$€£\-+=%()/@#&]', '', cleaned)
        
        # Limit text length to prevent abuse
        max_length = 10000  # 10KB limit
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length] + "..."
            logger.warning(f"Input text truncated to {max_length} characters")
        
        return cleaned
        
# Singleton instance
utility_service = UtilityService()