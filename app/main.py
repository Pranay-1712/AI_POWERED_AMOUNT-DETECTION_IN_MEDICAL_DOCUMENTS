# app/main.py
"""
Main FastAPI application for Medical Amount Detection
Implements the complete 4-step pipeline as per problem statement
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
import os
import io
from datetime import datetime
from typing import Union, Optional
from dotenv import load_dotenv

# Import our services and models
from .models import (
    FinalOutput, ErrorResponse, TextRequest, HealthResponse, APIInfo,
    OCROutput, NormalizationOutput, ClassificationOutput, ResponseModel,
    ClassificationRequest, NormalizationRequest
)
from .ocr_service import ocr_service
from .normalization import normalization_service
from .classification import GeminiClassificationService
from .utils import utility_service

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Medical Amount Detection API",
    description="AI-powered service that extracts financial amounts from medical bills and receipts",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for services
classification_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global classification_service
    
    # Get Gemini API key instead of OpenAI
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        logger.error("GEMINI_API_KEY environment variable not set!")
        raise ValueError("Gemini API key is required")
    
    # Initialize with Gemini classification service
    classification_service = GeminiClassificationService(gemini_api_key)
    
    logger.info("Medical Amount Detection API started successfully with Gemini")

# Dependency to get classification service
# Update the type hint (CHANGED)
def get_classification_service():
    """Dependency to inject classification service"""
    if classification_service is None:
        raise HTTPException(status_code=500, detail="Classification service not initialized")
    return classification_service

# Helper function to validate file upload
def validate_uploaded_file(file: UploadFile) -> None:
    """Validate uploaded file"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file size (10MB limit)
    max_size = 10 * 1024 * 1024  # 10MB
    if hasattr(file.file, 'seek') and hasattr(file.file, 'tell'):
        file.file.seek(0, 2)  # Seek to end
        size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        
        if size > max_size:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB")
    
    # Check file type
    allowed_types = ['image/jpeg', 'image/png', 'image/jpg', 'application/pdf']
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed types: {allowed_types}"
        )

# Main API endpoint - Extract amounts from medical documents
@app.post("/extract-amounts", response_model=ResponseModel)
async def extract_amounts(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    classification_svc: GeminiClassificationService = Depends(get_classification_service)
):
    """
    Extract and classify amounts from medical documents.
    
    **Input Options:**
    - Upload an image file (JPEG, PNG, PDF)
    - Provide text directly via form data
    - Provide text via JSON body (if no file)
    
    **Processing Pipeline:**
    1. OCR/Text Extraction
    2. Numeric Normalization  
    3. Context Classification
    4. Final Structured Output
    
    **Returns:**
    - Success: Structured JSON with classified amounts
    - Error: Error response with specific reason
    """
    
    try:
        logger.info("Processing extract-amounts request")
        processing_start_time = datetime.now()
        
        # === Input Validation ===
        if not file and not text:
            return utility_service.create_error_response(
                "error", 
                "No input provided. Please provide either a file or text."
            )
        
        # === Step 1: OCR/Text Extraction ===
        logger.info("Step 1: Starting OCR/Text Extraction")
        
        if file:
            # Validate uploaded file
            validate_uploaded_file(file)
            
            # Read file data
            file_data = await file.read()
            logger.info(f"Processing uploaded file: {file.filename} ({len(file_data)} bytes)")
            
            # Process with OCR
            ocr_result = ocr_service.process_input(image_data=file_data)
        
        else:
            # Process direct text input
            text = utility_service.sanitize_text_input(text)
            logger.info(f"Processing direct text input ({len(text)} characters)")
            
            ocr_result = ocr_service.process_input(text=text)
        
        logger.info(f"Step 1 completed: {len(ocr_result.raw_tokens)} tokens extracted, confidence: {ocr_result.confidence}")
        
        # === Guardrail Check 1: No amounts found ===
        if not ocr_result.raw_tokens:
            return utility_service.create_error_response(
                "no_amounts_found",
                "No numeric values detected in the document"
            )
        
        # === Guardrail Check 2: Low OCR confidence ===
        if ocr_result.confidence < 0.3:
            return utility_service.create_error_response(
                "low_confidence",
                "Document too noisy or unclear for reliable extraction"
            )
        
        # === Step 2: Normalization ===
        logger.info("Step 2: Starting Normalization")
        
        normalization_result = normalization_service.process_tokens(ocr_result.raw_tokens)
        
        logger.info(f"Step 2 completed: {len(normalization_result.normalized_amounts)} amounts normalized, confidence: {normalization_result.normalization_confidence}")
        
        # === Guardrail Check 3: Normalization failed ===
        if not normalization_result.normalized_amounts:
            return utility_service.create_error_response(
                "normalization_failed",
                "Could not normalize any detected amounts"
            )
        
        # === Step 3: Classification by Context ===
        logger.info("Step 3: Starting Classification")
        
        # Get original text for context
        if file:
            # Re-extract text for classification context (could be optimized)
            original_text, _ = ocr_service.extract_text_from_image(file_data)
        else:
            original_text = text
        
        classification_result = classification_svc.process_amounts(
            original_text, 
            normalization_result.normalized_amounts
        )
        
        logger.info(f"Step 3 completed: {len(classification_result.amounts)} amounts classified, confidence: {classification_result.confidence}")
        
        # === Step 4: Final Output Generation ===
        logger.info("Step 4: Generating Final Output")
        
        final_output = utility_service.generate_final_output(
            original_text,
            classification_result.amounts,
            ocr_result.currency_hint
        )
        
        # Log processing time
        processing_time = (datetime.now() - processing_start_time).total_seconds()
        logger.info(f"Step 4 completed: Final output generated in {processing_time:.2f}s")
        
        # === Success Response ===
        logger.info(f"Request completed successfully: {len(final_output.amounts)} amounts extracted")
        
        return final_output
        
    except HTTPException:
        # Re-raise HTTP exceptions (like file validation errors)
        raise
        
    except Exception as e:
        # Handle any other errors
        logger.error(f"Unexpected error during processing: {str(e)}", exc_info=True)
        return utility_service.create_error_response(
            "error",
            f"Internal processing error: {str(e)}"
        )

# Alternative endpoint for JSON text input
@app.post("/extract-amounts-json", response_model=ResponseModel)
async def extract_amounts_json(
    request: TextRequest,
    classification_svc: GeminiClassificationService = Depends(get_classification_service)
):
    """
    Extract amounts from text provided via JSON body.
    Alternative to the form-data endpoint for easier programmatic access.
    """
    return await extract_amounts(
        file=None,
        text=request.text,
        classification_svc=classification_svc
    )

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify API status and dependencies.
    """
    try:
        # Check if services are initialized
        dependencies = {
            "ocr_service": "ready" if ocr_service else "not_ready",
            "normalization_service": "ready" if normalization_service else "not_ready", 
            "classification_service": "ready" if classification_service else "not_ready",
            "utility_service": "ready" if utility_service else "not_ready"
        }
        
        # Check OpenAI API key
        gemini_key_status = "configured" if os.getenv("GEMINI_API_KEY") else "missing"
        dependencies["gemini_api_key"] = gemini_key_status
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            dependencies=dependencies
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# API information endpoint
@app.get("/", response_model=APIInfo)
async def root():
    """
    Root endpoint providing API information and available endpoints.
    """
    return APIInfo()

# Debug endpoints for testing individual steps (useful for development)
@app.post("/debug/step1-ocr", response_model=OCROutput)
async def debug_step1_ocr(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None)
):
    """Debug endpoint to test Step 1 (OCR) only"""
    try:
        if file:
            file_data = await file.read()
            return ocr_service.process_input(image_data=file_data)
        elif text:
            return ocr_service.process_input(text=text)
        else:
            raise HTTPException(status_code=400, detail="Provide file or text")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/debug/step2-normalization", response_model=NormalizationOutput)
async def debug_step2_normalization(request: NormalizationRequest):
    """Debug endpoint to test Step 2 (Normalization) only"""
    try:
        return normalization_service.process_tokens(request.tokens)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/debug/step3-classification", response_model=ClassificationOutput)
async def debug_step3_classification(
    request: ClassificationRequest,
    classification_svc: GeminiClassificationService = Depends(get_classification_service)
):
    """Debug endpoint to test Step 3 (Classification) only"""
    try:
        return classification_svc.process_amounts(request.text, request.amounts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions"""
    logger.error(f"ValueError: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={"status": "error", "reason": str(exc)}
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    logger.error(f"HTTPException: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "reason": exc.detail}
    )

# Custom middleware for request logging
@app.middleware("http")
async def log_requests(request, call_next):
    """Log all incoming requests"""
    start_time = datetime.now()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Response: {response.status_code} - {process_time:.2f}s")
    
    return response

# Run the application (for development)
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )