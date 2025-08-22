from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import base64
import json
from typing import Optional, List, Union
import uvicorn
import os
import logging

# Import FGSM functions from existing fgsm.py
from fgsm import fgsm_attack, targeted_fgsm_attack

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="FGSM Adversarial Attack API",
    description="RESTful API for generating adversarial examples using Fast Gradient Sign Method (FGSM)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple CNN model (you can replace this with your own model)
class SimpleCNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Global model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNNModel().to(device)
model.eval()

# Data preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

# Pydantic models for request/response
class FGSMAttackRequest(BaseModel):
    """Request model for FGSM attack"""
    image_data: str = Field(..., description="Base64 encoded image data (with or without data URL prefix)")
    true_label: int = Field(..., description="True label of the input image", ge=0, le=9)
    epsilon: float = Field(default=0.1, description="Perturbation strength (0.0 to 1.0)", ge=0.0, le=1.0)
    attack_type: str = Field(default="untargeted", description="Attack type: 'untargeted' or 'targeted'", pattern="^(untargeted|targeted)$")
    target_label: Optional[int] = Field(default=None, description="Target label for targeted attack (0-9)", ge=0, le=9)

class BatchFGSMRequest(BaseModel):
    """Request model for batch FGSM attacks"""
    images_data: List[str] = Field(..., description="List of base64 encoded images")
    true_labels: List[int] = Field(..., description="List of true labels corresponding to images")
    epsilon: float = Field(default=0.1, description="Perturbation strength", ge=0.0, le=1.0)
    attack_type: str = Field(default="untargeted", description="Attack type", pattern="^(untargeted|targeted)$")
    target_labels: Optional[List[int]] = Field(default=None, description="Target labels for targeted attacks")

class AttackResponse(BaseModel):
    """Response model for FGSM attack"""
    success: bool = Field(..., description="Whether the API call was successful")
    attack_success: bool = Field(..., description="Whether the adversarial attack was successful")
    message: str = Field(..., description="Status message")
    original_prediction: int = Field(..., description="Model's prediction on original image")
    adversarial_prediction: int = Field(..., description="Model's prediction on adversarial image")
    original_confidence: float = Field(..., description="Confidence score for original prediction")
    adversarial_confidence: float = Field(..., description="Confidence score for adversarial prediction")
    epsilon_used: float = Field(..., description="Epsilon value used for the attack")
    perturbation_norm: float = Field(..., description="L2 norm of the perturbation")
    adversarial_image: str = Field(..., description="Base64 encoded adversarial image")

class BatchAttackResponse(BaseModel):
    """Response model for batch FGSM attacks"""
    success: bool = Field(..., description="Whether the batch processing was successful")
    total_images: int = Field(..., description="Total number of images processed")
    successful_attacks: int = Field(..., description="Number of successful attacks")
    attack_success_rate: float = Field(..., description="Overall attack success rate")
    results: List[AttackResponse] = Field(..., description="Individual attack results")

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="API status")
    device: str = Field(..., description="Computing device being used")
    cuda_available: bool = Field(..., description="Whether CUDA is available")
    model_loaded: bool = Field(..., description="Whether the model is loaded successfully")

# Utility functions
def decode_base64_image(image_data: str) -> torch.Tensor:
    """
    Decode base64 image data and convert to tensor
    
    Args:
        image_data: Base64 encoded image string
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    try:
        # Remove data URL prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Apply transformations
        tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return tensor.to(device)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

def encode_tensor_to_base64(tensor: torch.Tensor) -> str:
    """
    Encode tensor as base64 image
    
    Args:
        tensor: Image tensor to encode
        
    Returns:
        str: Base64 encoded image with data URL prefix
    """
    try:
        # Convert tensor to PIL image
        tensor_cpu = tensor.squeeze().cpu().clamp(0, 1)
        image_array = (tensor_cpu.numpy() * 255).astype(np.uint8)
        image = Image.fromarray(image_array, mode='L')
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/png;base64,{image_b64}"
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error encoding image: {str(e)}")

def get_model_prediction(tensor: torch.Tensor):
    """
    Get model prediction and confidence score
    
    Args:
        tensor: Input tensor
        
    Returns:
        tuple: (prediction, confidence)
    """
    try:
        with torch.no_grad():
            output = model(tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
            return prediction.item(), confidence.item()
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction error: {str(e)}")

def validate_attack_request(request: FGSMAttackRequest):
    """Validate attack request parameters"""
    if request.attack_type == "targeted" and request.target_label is None:
        raise HTTPException(
            status_code=400, 
            detail="target_label is required for targeted attacks"
        )
    
    if request.attack_type == "targeted" and request.target_label == request.true_label:
        raise HTTPException(
            status_code=400,
            detail="target_label cannot be the same as true_label for targeted attacks"
        )

# API Endpoints
@app.get("/", summary="API Information")
async def root():
    """Get basic API information"""
    return {
        "message": "FGSM Adversarial Attack API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "attack": "/attack",
            "batch_attack": "/batch-attack",
            "file_attack": "/attack/file",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }

@app.get("/health", response_model=HealthResponse, summary="Health Check")
async def health_check():
    """Check API health and system status"""
    return HealthResponse(
        status="healthy",
        device=str(device),
        cuda_available=torch.cuda.is_available(),
        model_loaded=model is not None
    )

@app.post("/attack", response_model=AttackResponse, summary="Generate Adversarial Example")
async def fgsm_attack_endpoint(request: FGSMAttackRequest):
    """
    Generate adversarial example using FGSM attack
    
    This endpoint accepts an image, true label, and attack parameters,
    then returns the adversarial example along with attack success metrics.
    """
    try:
        # Validate request
        validate_attack_request(request)
        
        # Decode input image
        image_tensor = decode_base64_image(request.image_data)
        
        # Get original prediction
        orig_pred, orig_conf = get_model_prediction(image_tensor)
        
        # Prepare labels
        true_label_tensor = torch.tensor([request.true_label], device=device)
        
        # Perform FGSM attack
        if request.attack_type == "targeted":
            target_label_tensor = torch.tensor([request.target_label], device=device)
            adversarial_tensor = targeted_fgsm_attack(
                model, image_tensor, target_label_tensor, request.epsilon, device
            )
        else:
            adversarial_tensor = fgsm_attack(
                model, image_tensor, true_label_tensor, request.epsilon, device
            )
        
        # Get adversarial prediction
        adv_pred, adv_conf = get_model_prediction(adversarial_tensor)
        
        # Calculate perturbation metrics
        perturbation = adversarial_tensor - image_tensor
        perturbation_norm = torch.norm(perturbation).item()
        
        # Determine attack success
        if request.attack_type == "targeted":
            attack_success = (adv_pred == request.target_label)
            success_msg = f"Targeted attack {'successful' if attack_success else 'failed'}"
        else:
            attack_success = (adv_pred != request.true_label) and (orig_pred == request.true_label)
            success_msg = f"Untargeted attack {'successful' if attack_success else 'failed'}"
        
        # Encode adversarial image
        adversarial_image_b64 = encode_tensor_to_base64(adversarial_tensor)
        
        logger.info(f"FGSM attack completed: {success_msg}")
        
        return AttackResponse(
            success=True,
            attack_success=attack_success,
            message=success_msg,
            original_prediction=orig_pred,
            adversarial_prediction=adv_pred,
            original_confidence=orig_conf,
            adversarial_confidence=adv_conf,
            epsilon_used=request.epsilon,
            perturbation_norm=perturbation_norm,
            adversarial_image=adversarial_image_b64
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"FGSM attack error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/batch-attack", response_model=BatchAttackResponse, summary="Batch Adversarial Attack")
async def batch_fgsm_attack_endpoint(request: BatchFGSMRequest):
    """
    Generate adversarial examples for multiple images using FGSM
    
    This endpoint processes multiple images in a single request and returns
    individual results along with overall success statistics.
    """
    try:
        # Validate batch request
        if len(request.images_data) != len(request.true_labels):
            raise HTTPException(
                status_code=400,
                detail="Number of images must match number of true labels"
            )
        
        if request.attack_type == "targeted":
            if not request.target_labels or len(request.target_labels) != len(request.images_data):
                raise HTTPException(
                    status_code=400,
                    detail="Target labels must be provided for all images in targeted attacks"
                )
        
        results = []
        successful_attacks = 0
        
        # Process each image
        for i, (image_data, true_label) in enumerate(zip(request.images_data, request.true_labels)):
            try:
                # Create individual request
                individual_request = FGSMAttackRequest(
                    image_data=image_data,
                    true_label=true_label,
                    epsilon=request.epsilon,
                    attack_type=request.attack_type,
                    target_label=request.target_labels[i] if request.target_labels else None
                )
                
                # Process individual attack
                result = await fgsm_attack_endpoint(individual_request)
                results.append(result)
                
                if result.attack_success:
                    successful_attacks += 1
                    
            except Exception as e:
                # Create failed result for this image
                failed_result = AttackResponse(
                    success=False,
                    attack_success=False,
                    message=f"Failed to process image {i+1}: {str(e)}",
                    original_prediction=-1,
                    adversarial_prediction=-1,
                    original_confidence=0.0,
                    adversarial_confidence=0.0,
                    epsilon_used=request.epsilon,
                    perturbation_norm=0.0,
                    adversarial_image=""
                )
                results.append(failed_result)
        
        # Calculate success rate
        attack_success_rate = successful_attacks / len(request.images_data) if request.images_data else 0.0
        
        logger.info(f"Batch FGSM completed: {successful_attacks}/{len(request.images_data)} successful")
        
        return BatchAttackResponse(
            success=True,
            total_images=len(request.images_data),
            successful_attacks=successful_attacks,
            attack_success_rate=attack_success_rate,
            results=results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch FGSM error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/attack/file", response_model=AttackResponse, summary="File Upload Attack")
async def file_attack_endpoint(
    file: UploadFile = File(..., description="Image file to attack"),
    true_label: int = Form(..., description="True label of the image", ge=0, le=9),
    epsilon: float = Form(0.1, description="Perturbation strength", ge=0.0, le=1.0),
    attack_type: str = Form("untargeted", description="Attack type (untargeted/targeted)", pattern="^(untargeted|targeted)$"),
    target_label: Optional[int] = Form(None, description="Target label for targeted attack", ge=0, le=9)
):
    """
    Upload an image file and generate adversarial example
    
    This endpoint allows direct file upload instead of base64 encoding.
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content
        contents = await file.read()
        
        # Convert to base64
        image_b64 = base64.b64encode(contents).decode('utf-8')
        
        # Create request object
        attack_request = FGSMAttackRequest(
            image_data=image_b64,
            true_label=true_label,
            epsilon=epsilon,
            attack_type=attack_type,
            target_label=target_label
        )
        
        # Process attack
        return await fgsm_attack_endpoint(attack_request)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File attack error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "status_code": 500
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logger.info("FGSM Adversarial Attack API starting...")
    logger.info(f"Device: {device}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info("API ready to serve requests")

if __name__ == "__main__":
    print("🚀 Starting FGSM Adversarial Attack API")
    print("=" * 60)
    print(f"📱 Device: {device}")
    print(f"🔥 CUDA Available: {torch.cuda.is_available()}")
    print("📚 Documentation: http://localhost:8000/docs")
    print("📖 Alternative docs: http://localhost:8000/redoc")
    print("🏥 Health check: http://localhost:8000/health")
    print("=" * 60)
    
    uvicorn.run(
        "app_fgsm:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )