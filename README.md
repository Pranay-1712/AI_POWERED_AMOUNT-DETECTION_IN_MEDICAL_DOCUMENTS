# Medical Amount Detection API

AI-powered service that extracts and classifies financial amounts from medical bills and receipts.

## Architecture

4-step processing pipeline:

```
Input → OCR → Normalization → Classification → Final Output
```

- **Step 1: OCR** - Extract text and numeric tokens from images/text
- **Step 2: Normalization** - Clean OCR errors and normalize amounts  
- **Step 3: Classification** - Use Gemini AI to classify amounts by context
- **Step 4: Final Output** - Generate structured JSON with provenance

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Tesseract OCR
**macOS:** `brew install tesseract`  
**Ubuntu:** `sudo apt-get install tesseract-ocr`  
**Windows:** Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)

### 3. Configure Environment
Create `.env` file:
```env
GEMINI_API_KEY=gemini_api_key
```

### 4. Run Application
```bash
python3 run.py
```

API available at `http://localhost:8000`

## API Usage

### Main Endpoint
**POST** `/extract-amounts`

**Text Input (JSON):**
```bash
curl -X POST "http://localhost:8000/extract-amounts-json" \
  -H "Content-Type: application/json" \
  -d '{"text": "Total Amount: Rs 4,000 | Advance: Rs 2,000 | Balance Due: Rs 2,000"}'
```

**Image Upload:**
```bash
curl -X POST "http://localhost:8000/extract-amounts" \
  -F "file=@medical_bill.jpeg"
```

### Response Format
```json
{
  "currency": "INR",
  "amounts": [
    {
      "type": "total_bill",
      "value": 4000,
      "source": "text: 'Total Amount: Rs 4,000'",
      "name": "Total Amount"
    },
    {
      "type": "paid",
      "value": 2000,
      "source": "text: 'Advance: Rs 2,000'",
      "name": "Advance Payment"
    },
    {
      "type": "due",
      "value": 2000,
      "source": "text: 'Balance Due: Rs 2,000'",
      "name": "Balance Due"
    }
  ],
  "status": "ok"
}
```
## Postman Collection

Import the Postman collection from `postman/Medical_Amount_Detection.postman_collection.json` to test all endpoints.

### Collection Structure:
- **🏥 Main Endpoints**
  - Health Check
  - Extract Amounts - JSON Text
  - Extract Amounts - Image Upload

- **🔧 Debug Endpoints**
  - Step 1: OCR (Text/Image)
  - Step 2: Normalization
  - Step 3: Classification

- **🧪 Test Cases**
  - Simple Consultation - Text
  - Hospital Bill - Text
  - Consultation Bill - Image
  - Hospital Bill - Image
  - Pharmacy Bill - Image

### Quick Postman Setup:
1. Open Postman
2. Click "Import" → "Upload Files"
3. Select `postman/Medical_Amount_Detection.postman_collection.json`
4. Start testing with "Health Check" endpoint
5. Try "Extract Amounts - JSON Text" with sample data
6. Test image uploads using `test_data/test_image_1.png` (or test_image_2.png, test_image_3.png)
Common error statuses: `no_amounts_found`, `low_confidence`, `normalization_failed`, `error`
