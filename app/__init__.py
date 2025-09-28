# app/__init__.py
"""
Medical Amount Detection API Package
AI-powered service for extracting financial amounts from medical documents
"""

__version__ = "1.0.0"
__author__ = "Medical Amount Detection Team"
__description__ = "AI-powered service that extracts financial amounts from medical bills and receipts"

# Import main components for easy access
from .main import app
from .models import FinalOutput, ErrorResponse, OCROutput, NormalizationOutput, ClassificationOutput
from .ocr_service import ocr_service
from .normalization import normalization_service
from .utils import utility_service

__all__ = [
    "app",
    "FinalOutput", 
    "ErrorResponse",
    "OCROutput",
    "NormalizationOutput", 
    "ClassificationOutput",
    "ocr_service",
    "normalization_service",
    "utility_service"
]