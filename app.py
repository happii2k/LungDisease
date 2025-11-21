# FastAPI application for X-ray image classification

import io
import logging
import torch
from src.dl.CustomNN import Net
from src.constant import PREDICTION_LABEL
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends , Request
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import time
from src.logger import logging
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Add custom model to safe globals for PyTorch 2.6+
torch.serialization.add_safe_globals([Net])

# Initialize FastAPI
app = FastAPI(
    title="Lung Disease X-Ray Classification API",
    description="API for predicting lung diseases from X-ray images",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# File size limit (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

# Allowed image formats
ALLOWED_FORMATS = {"image/jpeg", "image/jpg", "image/png"}


# File size validation dependency


# Initialize and load model
try:
    logging.info("Loading model...")
    model = Net().to(device)
    model.load_state_dict(torch.load('artifacts/model_training/xray_model.pth', map_location=torch.device('cpu')))
    model.eval()
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Failed to load model: {str(e)}", exc_info=True)
    raise


# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Root endpoint - Homepage"""
    api_info = {
        "name": "Lung Disease X-Ray Classification API",
        "version": "1.0.0",
        "status": "running",
        "device": str(device),
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }
    return templates.TemplateResponse("home.html", {"request": request, "api": api_info})


@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    """Render the prediction page"""
    return templates.TemplateResponse("index.html", {"request": request})



@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    """
    Predict lung disease from X-ray image
    
    Args:
        file: X-ray image file (JPEG/PNG, max 10MB)
    
    Returns:
        JSON with prediction label, index, and confidence score
    """
    start_time = time.time()
    
    try:
        # Validate file type
        if file.content_type not in ALLOWED_FORMATS:
            logging.warning(f"Invalid file type: {file.content_type}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_FORMATS)}"
            )
        
        logging.info(f"Processing file: {file.filename}, type: {file.content_type}")
        
        # Read and validate image
        contents = await file.read()
        
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            logging.info(f"Image loaded successfully. Size: {image.size}")
        except Exception as e:
            logging.error(f"Failed to open image: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Transform image
        input_tensor = transform(image).unsqueeze(0).to(device)
        logging.debug(f"Input tensor shape: {input_tensor.shape}")
        
        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            prediction_index = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction_index].item()
        
        prediction_label = PREDICTION_LABEL.get(prediction_index, "Unknown")
        
        processing_time = time.time() - start_time
        
        logging.info(
            f"Prediction: {prediction_label} (index: {prediction_index}), "
            f"Confidence: {confidence:.4f}, Time: {processing_time:.3f}s"
        )
        
        return {
            "success": True,
            "prediction_index": prediction_index,
            "prediction_label": prediction_label,
            "confidence": round(confidence, 4),
            "processing_time_seconds": round(processing_time, 3),
            "all_probabilities": {
                PREDICTION_LABEL[i]: round(probabilities[0][i].item(), 4)
                for i in range(len(PREDICTION_LABEL))
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during prediction: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors"""
    logging.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test model is loaded
        if model is None:
            raise Exception("Model not loaded")
        
        return {
            "status": "healthy",
            "model": "loaded",
            "device": str(device),
            "timestamp": time.time()
        }
    except Exception as e:
        logging.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
