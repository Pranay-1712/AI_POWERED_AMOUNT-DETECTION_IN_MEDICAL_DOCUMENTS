# app/models.py
"""
Pydantic models for request/response validation
Following the exact JSON schemas from problem statement
"""
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union

class OCROutput(BaseModel):
    """Output from Step 1 - OCR/Text Extraction"""
    raw_tokens: List[str] = Field(..., description="Raw numeric tokens extracted")
    currency_hint: str = Field(..., description="Detected currency hint")
    confidence: float = Field(..., ge=0.0, le=1.0, description="OCR confidence score")

    class Config:
        json_schema_extra = {
            "example": {
                "raw_tokens": ["1200", "1000", "200", "10%"],
                "currency_hint": "INR",
                "confidence": 0.74
            }
        }

# Step 2: Normalization Models
class NormalizationOutput(BaseModel):
    """Output from Step 2 - Normalization"""
    normalized_amounts: List[float] = Field(..., description="Normalized numeric amounts")
    normalization_confidence: float = Field(..., ge=0.0, le=1.0, description="Normalization confidence")

    class Config:
        json_schema_extra = {
            "example": {
                "normalized_amounts": [1200, 1000, 200],
                "normalization_confidence": 0.82
            }
        }

# Step 3: Classification Models
class ClassifiedAmount(BaseModel):
    """Individual classified amount"""
    type: str = Field(..., description="Type of amount")
    value: float = Field(..., description="Numeric value")
    context: Optional[str] = Field(None, description="Context explanation")
    name: Optional[str] = Field(None, description="Specific name or description of the item/service")

class ClassificationOutput(BaseModel):
    """Output from Step 3 - Classification by Context"""
    amounts: List[ClassifiedAmount] = Field(..., description="Classified amounts")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")

    class Config:
        json_schema_extra = {
            "example": {
                "amounts": [
                    {"type": "total_bill", "value": 1200},
                    {"type": "paid", "value": 1000},
                    {"type": "due", "value": 200}
                ],
                "confidence": 0.80
            }
        }

# Step 4: Final Output Models
class AmountInfo(BaseModel):
    """Individual amount in final output"""
    type: str = Field(..., description="Type of amount")
    value: float = Field(..., description="Numeric value")
    source: str = Field(..., description="Source text with provenance")
    name: Optional[str] = Field(None, description="Specific name or description of the item/service")

    @validator('type')
    def validate_type(cls, v):
        valid_types = [
            'total_bill', 'paid', 'due', 'discount', 'tax',
            'consultation', 'medicine', 'test', 'other'
        ]
        if v not in valid_types:
            return 'other'  # Default to 'other' if invalid type
        return v

class FinalOutput(BaseModel):
    """Final output from Step 4 - Structured JSON with provenance"""
    currency: str = Field(..., description="Detected currency")
    amounts: List[AmountInfo] = Field(..., description="Classified amounts with provenance")
    status: str = Field(..., description="Processing status")

    @validator('currency')
    def validate_currency(cls, v):
        if v not in ['INR', 'USD', 'EUR']:
            return 'INR'  # Default to INR
        return v

    @validator('status')
    def validate_status(cls, v):
        if v not in ['ok', 'no_amounts_found', 'low_confidence', 'normalization_failed', 'error']:
            return 'ok'
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "currency": "INR",
                "amounts": [
                    {
                        "type": "total_bill",
                        "value": 1200,
                        "source": "text: 'Total: INR 1200'",
                        "name": "Total Amount"
                    },
                    {
                        "type": "paid",
                        "value": 1000,
                        "source": "text: 'Paid: 1000'",
                        "name": "Advance Payment"
                    },
                    {
                        "type": "due",
                        "value": 200,
                        "source": "text: 'Due: 200'",
                        "name": "Balance Due"
                    },
                    {
                        "type": "medicine",
                        "value": 150,
                        "source": "text: 'Paracetamol 200mg: Rs 150'",
                        "name": "Paracetamol 200mg"
                    }
                ],
                "status": "ok"
            }
        }

# Guardrail/Error Response Models
class ErrorResponse(BaseModel):
    """Error response for guardrail conditions"""
    status: str = Field(..., description="Error status")
    reason: str = Field(..., description="Error reason/description")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "no_amounts_found",
                "reason": "document too noisy"
            }
        }

# Request Models
class TextRequest(BaseModel):
    """Request model for text input"""
    text: str = Field(..., min_length=1, description="Input text to process")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Total: INR 1200 | Paid: 1000 | Due: 200 | Discount: 10%"
            }
        }

class ClassificationRequest(BaseModel):
    """Request model for classification debug endpoint"""
    text: str = Field(..., min_length=1, description="Input text to classify")
    amounts: List[float] = Field(..., description="List of amounts to classify")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Total: INR 1200 | Paid: 1000 | Due: 200",
                "amounts": [1200, 1000, 200]
            }
        }

class NormalizationRequest(BaseModel):
    """Request model for normalization debug endpoint"""
    tokens: List[str] = Field(..., description="List of tokens to normalize")

    class Config:
        json_schema_extra = {
            "example": {
                "tokens": ["1200", "1000", "200", "10%"]
            }
        }

# Response Models Union
ResponseModel = Union[FinalOutput, ErrorResponse]

# Health check model
class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "healthy"
    timestamp: str
    version: str = "1.0.0"
    dependencies: Dict[str, str] = {}

# API Info model
class APIInfo(BaseModel):
    """API information response"""
    name: str = "Medical Amount Detection API"
    version: str = "1.0.0"
    description: str = "AI-powered service for extracting financial amounts from medical documents"
    endpoints: Dict[str, str] = {
        "POST /extract-amounts": "Extract amounts from medical documents",
        "GET /health": "Health check endpoint",
        "GET /": "API information",
        "GET /docs": "Interactive API documentation"
    }
    supported_formats: List[str] = ["JPEG", "PNG", "PDF", "Text"]
    max_file_size: str = "10MB"