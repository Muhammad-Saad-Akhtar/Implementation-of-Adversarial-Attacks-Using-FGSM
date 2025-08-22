from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
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
from typing import Optional, List
import uvicorn
import os

# Import our FGSM implementation
from fgsm import fgsm_attack, targeted_fgsm_attack

app = FastAPI(
    title="FGSM Adversarial Attack API",
    description="RESTful API for generating adversarial examples using Fast Gradient Sign Method (FGSM)",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Simple MNIST model for demonstration
class SimpleMNISTModel(nn.Module):
    def __init__(self):
        super(SimpleMNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

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

# Global model instance (in production, you'd load this from a saved checkpoint)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleMNISTModel().to(device)
model.eval()  # Set to evaluation mode

# Data transformation for preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

# Request models
class FGSMRequest(BaseModel):
    """Request model for FGSM attack"""
    image_data: str = Field(..., description="Base64 encoded image data")
    true_label: int = Field(..., description="True label of the image", ge=0, le=9)
    epsilon: float = Field(default=0.1, description="Perturbation strength", ge=0.0, le=1.0)
    attack_type: str = Field(default="untargeted", description="Attack type: 'untargeted' or 'targeted'")
    target_label: Optional[int] = Field(default=None, description="Target label for targeted attack", ge=0, le=9)

class BatchFGSMRequest(BaseModel):
    """Request model for batch FGSM attack"""
    images_data: List[str] = Field(..., description="List of base64 encoded image data")
    true_labels: List[int] = Field(..., description="List of true labels")
    epsilon: float = Field(default=0.1, description="Perturbation strength", ge=0.0, le=1.0)
    attack_type: str = Field(default="untargeted", description="Attack type")
    target_labels: Optional[List[int]] = Field(default=None, description="Target labels for targeted attack")

# Response models
class FGSMResponse(BaseModel):
    """Response model for FGSM attack"""
    success: bool
    message: str
    original_prediction: int
    adversarial_prediction: int
    original_confidence: float
    adversarial_confidence: float
    adversarial_image: str  # Base64 encoded
    attack_success: bool
    epsilon_used: float
    perturbation_norm: float

class BatchFGSMResponse(BaseModel):
    """Response model for batch FGSM attack"""
    success: bool
    message: str
    results: List[FGSMResponse]
    overall_attack_success_rate: float
    total_processed: int

# Utility functions
def decode_base64_image(image_data: str) -> torch.Tensor:
    """Decode base64 image data and convert to tensor"""
    try:
        # Remove data URL prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Transform to tensor
        tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return tensor.to(device)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

def encode_tensor_to_base64(tensor: torch.Tensor) -> str:
    """Encode tensor as base64 image"""
    try:
        # Convert to PIL image
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
    """Get model prediction and confidence"""
    with torch.no_grad():
        output = model(tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
        return prediction.item(), confidence.item()

# API endpoints
@app.get("/", response_class=HTMLResponse)
async def root(request):
    """Serve the frontend interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": str(device),
        "model_loaded": model is not None
    }

@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    return {
        "model_type": "SimpleMNISTModel",
        "input_shape": [1, 28, 28],
        "num_classes": 10,
        "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters())
    }

@app.post("/attack", response_model=FGSMResponse)
async def fgsm_attack_endpoint(request: FGSMRequest):
    """Generate adversarial example using FGSM attack"""
    try:
        # Decode image
        image_tensor = decode_base64_image(request.image_data)
        
        # Get original prediction
        orig_pred, orig_conf = get_model_prediction(image_tensor)
        
        # Prepare labels
        true_label = torch.tensor([request.true_label], device=device)
        
        # Generate adversarial example
        if request.attack_type == "targeted":
            if request.target_label is None:
                raise HTTPException(status_code=400, detail="Target label required for targeted attack")
            
            target_label = torch.tensor([request.target_label], device=device)
            adversarial_tensor = targeted_fgsm_attack(model, image_tensor, target_label, request.epsilon, device)
        else:
            # Untargeted attack
            adversarial_tensor = fgsm_attack(model, image_tensor, true_label, request.epsilon, device)
        
        # Get adversarial prediction
        adv_pred, adv_conf = get_model_prediction(adversarial_tensor)
        
        # Calculate perturbation norm
        perturbation = adversarial_tensor - image_tensor
        perturbation_norm = torch.norm(perturbation).item()
        
        # Determine attack success
        if request.attack_type == "targeted":
            attack_success = (adv_pred == request.target_label)
        else:
            attack_success = (adv_pred != request.true_label) and (orig_pred == request.true_label)
        
        # Encode adversarial image
        adversarial_image_b64 = encode_tensor_to_base64(adversarial_tensor)
        
        return FGSMResponse(
            success=True,
            message="FGSM attack completed successfully",
            original_prediction=orig_pred,
            adversarial_prediction=adv_pred,
            original_confidence=orig_conf,
            adversarial_confidence=adv_conf,
            adversarial_image=adversarial_image_b64,
            attack_success=attack_success,
            epsilon_used=request.epsilon,
            perturbation_norm=perturbation_norm
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/attack/batch", response_model=BatchFGSMResponse)
async def batch_fgsm_attack_endpoint(request: BatchFGSMRequest):
    """Generate adversarial examples for multiple images using FGSM"""
    try:
        if len(request.images_data) != len(request.true_labels):
            raise HTTPException(status_code=400, detail="Number of images and labels must match")
        
        if request.attack_type == "targeted":
            if not request.target_labels or len(request.target_labels) != len(request.images_data):
                raise HTTPException(status_code=400, detail="Target labels required for targeted attack")
        
        results = []
        successful_attacks = 0
        
        for i, (image_data, true_label) in enumerate(zip(request.images_data, request.true_labels)):
            try:
                # Create individual request
                individual_request = FGSMRequest(
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
                # Add failed result
                results.append(FGSMResponse(
                    success=False,
                    message=f"Failed to process image {i}: {str(e)}",
                    original_prediction=-1,
                    adversarial_prediction=-1,
                    original_confidence=0.0,
                    adversarial_confidence=0.0,
                    adversarial_image="",
                    attack_success=False,
                    epsilon_used=request.epsilon,
                    perturbation_norm=0.0
                ))
        
        success_rate = successful_attacks / len(request.images_data) if request.images_data else 0.0
        
        return BatchFGSMResponse(
            success=True,
            message="Batch FGSM attack completed",
            results=results,
            overall_attack_success_rate=success_rate,
            total_processed=len(request.images_data)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/attack/file")
async def fgsm_attack_file_upload(
    file: UploadFile = File(...),
    true_label: int = 0,
    epsilon: float = 0.1,
    attack_type: str = "untargeted",
    target_label: Optional[int] = None
):
    """Upload image file and generate adversarial example"""
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Convert to base64
        image_b64 = base64.b64encode(contents).decode('utf-8')
        
        # Create request
        request = FGSMRequest(
            image_data=image_b64,
            true_label=true_label,
            epsilon=epsilon,
            attack_type=attack_type,
            target_label=target_label
        )
        
        # Process attack
        return await fgsm_attack_endpoint(request)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload error: {str(e)}")

# Custom exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )

if __name__ == "__main__":
    print("Starting FGSM Adversarial Attack API")
    print("="*50)
    print("API Documentation available at: http://localhost:8000/docs")
    print("Alternative docs at: http://localhost:8000/redoc")
    print("="*50)
    
    uvicorn.run(
        "app_fgsm:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )