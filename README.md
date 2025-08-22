# FGSM Adversarial Attack Implementation

This repository contains a comprehensive implementation of the Fast Gradient Sign Method (FGSM) adversarial attack, proposed by Goodfellow et al., along with Gaussian noise variants and a RESTful API with a modern web interface.

## 🚀 Quick Demo

1. **Start the backend server:**
   ```bash
   python app_fgsm.py
   ```

2. **Open the web interface:**
   - Open `index.html` in your browser
   - Upload an image and configure attack parameters
   - Generate adversarial examples with a single click!

## 📋 Table of Contents

- [Files Overview](#files-overview)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Web Interface](#web-interface)
- [API Usage](#api-usage)
- [Core Functions](#core-functions)
- [Evaluation Results](#evaluation-results)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## Files Overview

### Core Implementations
- **`fgsm.py`** - Original FGSM attack implementation
- **`fgsm_gaussian.py`** - Modified FGSM using Gaussian noise
- **`evaluate_attacks.py`** - Comprehensive evaluation script
- **`app_fgsm.py`** - FastAPI REST API for FGSM attacks
- **`index.html`** - Modern web interface for easy interaction
- **`train_model.py`** - Script to train the MNIST model
- **`test_api.py`** - API testing script

## Project Structure

```
Question_1/
├── 📁 Core Implementation
│   ├── fgsm.py                 # Original FGSM attack
│   ├── fgsm_gaussian.py        # Gaussian noise variants
│   └── evaluate_attacks.py     # Comprehensive evaluation
├── 🌐 Web Interface
│   ├── app_fgsm.py            # FastAPI backend server
│   ├── index.html             # Modern web frontend
│   └── test_api.py            # API testing utilities
├── 🤖 Model & Training
│   ├── train_model.py         # MNIST model training
│   └── models/                # Trained model storage
├── 📊 Data & Results
│   ├── data/MNIST/            # MNIST dataset
│   └── results/               # Generated plots and metrics
└── 📚 Documentation
    ├── README.md              # This file
    └── FGSM_Paper.pdf         # Original research paper
```

## Quick Start

### 1. System Requirements

- **Python**: 3.7 or higher
- **Memory**: At least 4GB RAM (8GB recommended)
- **Storage**: 500MB free space
- **GPU**: Optional (CUDA support for faster processing)

### 2. Install Dependencies

```bash
pip install torch torchvision matplotlib fastapi uvicorn pillow numpy pydantic
```

Or install from requirements.txt (if available):
```bash
pip install -r requirements.txt
```

### 2. Run Basic Evaluation

```python
# Run comprehensive evaluation
python evaluate_attacks.py
```

This will:
- Train a simple MNIST model
- Evaluate both FGSM and Gaussian attacks
- Generate visualization plots
- Save results to JSON files

### 3. Start the API Server

```bash
python app_fgsm.py
```

The API will be available at:
- **API Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

### 4. Use the Web Interface

Open `index.html` in your browser to access the user-friendly web interface with:
- 📁 Image upload and preview
- ⚙️ Attack parameter configuration
- 🎯 Real-time attack execution
- 📊 Visual results display
- 🖼️ Side-by-side comparison of original vs adversarial images

## Web Interface

The `index.html` file provides a modern, user-friendly web interface for interacting with the FGSM API.

### Features
- **🎨 Modern UI**: Clean, responsive design with gradient backgrounds
- **📁 Drag & Drop**: Easy image upload with preview
- **⚙️ Parameter Control**: Intuitive form for attack configuration
- **🔄 Real-time Processing**: Live status updates and loading indicators
- **📊 Rich Results**: Formatted output with confidence scores and metrics
- **🖼️ Visual Comparison**: Side-by-side display of original and adversarial images
- **🔗 API Integration**: Direct connection to FastAPI backend

### Usage
1. **Upload Image**: Select any image file (PNG, JPG, etc.)
2. **Configure Parameters**:
   - **True Label**: The correct label (0-9 for MNIST digits)
   - **Epsilon**: Perturbation strength (0.0-1.0)
   - **Attack Type**: Untargeted or targeted attack
   - **Target Label**: Desired misclassification (for targeted attacks)
3. **Execute Attack**: Click "Run FGSM Attack" to generate adversarial examples
4. **View Results**: See attack success, confidence scores, and visual comparison

### Supported Operations
- **Single Attack**: Generate adversarial example for one image
- **Batch Attack**: Process multiple images simultaneously
- **Health Check**: Verify backend connectivity
- **Clear Results**: Reset the interface

## Core Functions

### FGSM Attack (`fgsm.py`)

```python
from fgsm import fgsm_attack

# Generate adversarial examples
adversarial_examples = fgsm_attack(
    model=your_model,
    data=input_tensor,
    target=true_labels,
    epsilon=0.1
)
```

### Gaussian Attack (`fgsm_gaussian.py`)

```python
from fgsm_gaussian import fgsm_gaussian_attack

# Pure Gaussian noise attack
adversarial_examples = fgsm_gaussian_attack(
    model=your_model,
    data=input_tensor,
    target=true_labels,
    epsilon=0.1,
    std=1.0
)
```

## API Usage Examples

### Single Attack Request

```python
import requests
import base64

# Prepare image data
with open('image.png', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

# API request
response = requests.post('http://localhost:8000/attack', json={
    'image_data': image_data,
    'true_label': 5,
    'epsilon': 0.1,
    'attack_type': 'untargeted'
})

result = response.json()
print(f"Attack success: {result['attack_success']}")
print(f"Original prediction: {result['original_prediction']}")
print(f"Adversarial prediction: {result['adversarial_prediction']}")
```

### Batch Attack Request

```python
# Multiple images
response = requests.post('http://localhost:8000/attack/batch', json={
    'images_data': [image_data_1, image_data_2, image_data_3],
    'true_labels': [1, 2, 3],
    'epsilon': 0.1,
    'attack_type': 'untargeted'
})

batch_result = response.json()
print(f"Success rate: {batch_result['overall_attack_success_rate']}")
```

### File Upload Attack

```python
# Upload file directly
files = {'file': open('image.png', 'rb')}
data = {
    'true_label': 5,
    'epsilon': 0.1,
    'attack_type': 'untargeted'
}

response = requests.post('http://localhost:8000/attack/file', 
                        files=files, data=data)
```

## Evaluation Results

The evaluation script generates:

1. **JSON Results** - Detailed metrics for all epsilon values
2. **Visualization Plots** - Accuracy vs epsilon, attack success rates
3. **Adversarial Examples** - Visual comparison of original vs adversarial images

### Expected Output Structure

```
results/
├── attack_results_YYYYMMDD_HHMMSS.json
├── attack_comparison_YYYYMMDD_HHMMSS.png
└── adversarial_examples_YYYYMMDD_HHMMSS.png
```

### Sample Results Format

```json
{
  "clean_accuracy": 0.9856,
  "fgsm": {
    "0.1": {
      "original_accuracy": 0.9856,
      "adversarial_accuracy": 0.7834,
      "attack_success_rate": 0.2045
    }
  },
  "gaussian": {
    "0.1": {
      "pure_gaussian": {
        "original_accuracy": 0.9856,
        "adversarial_accuracy": 0.8923,
        "attack_success_rate": 0.0945
      }
    }
  }
}
```

## Key Features

### FGSM Implementation
- **Untargeted Attack**: Tries to change any correct prediction
- **Targeted Attack**: Tries to make model predict specific class
- **Batch Processing**: Efficient processing of multiple examples
- **Gradient Clipping**: Maintains valid pixel ranges

### Gaussian Variants
- **Pure Gaussian**: Random noise without gradient information
- **Gradient-Based Gaussian**: Uses gradient direction with Gaussian magnitude

### API Features
- **RESTful Interface**: Standard HTTP endpoints
- **Base64 Image Handling**: Easy integration with web applications
- **Batch Processing**: Handle multiple images efficiently
- **File Upload**: Direct file upload support
- **Comprehensive Responses**: Detailed attack results and metrics

## Performance Analysis

### Attack Effectiveness
- **FGSM**: Most effective, uses gradient information
- **Pure Gaussian**: Less effective, random perturbations
- **Gradient Gaussian**: Moderate effectiveness

### Computational Efficiency
- **FGSM**: Single forward + backward pass
- **Gaussian variants**: Forward pass only (faster)
- **API**: Optimized for real-time processing

## Advanced Usage

### Custom Model Integration

```python
# Use with your own model
your_model = YourCustomModel()
your_model.load_state_dict(torch.load('model.pth'))

# Apply FGSM
adversarial = fgsm_attack(your_model, data, labels, epsilon=0.1)
```

### Parameter Tuning

```python
# Test different epsilon values
epsilons = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
for eps in epsilons:
    accuracy = evaluate_attack(model, test_loader, eps, device)
    print(f"Epsilon {eps}: Accuracy {accuracy:.4f}")
```

### Robustness Testing

```python
# Comprehensive robustness evaluation
results = comprehensive_attack_evaluation(
    model=your_model,
    test_loader=test_data,
    device='cuda',
    output_dir='robustness_results'
)
```

## Mathematical Background

### FGSM Formula
The FGSM attack generates adversarial examples using:

```
x_adv = x + ε × sign(∇_x J(θ, x, y))
```

Where:
- `x`: Original input
- `x_adv`: Adversarial example
- `ε`: Perturbation magnitude (epsilon)
- `∇_x J(θ, x, y)`: Gradient of loss w.r.t. input
- `sign()`: Sign function

### Gaussian Variant
The Gaussian variant replaces the gradient sign with random noise:

```
x_adv = x + ε × N(0, σ²)
```

Where `N(0, σ²)` is Gaussian noise with standard deviation σ.

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size
   test_loader = DataLoader(dataset, batch_size=32)  # Instead of 1000
   ```

2. **Model Loading Errors**
   ```python
   # Ensure model is in eval mode
   model.eval()
   ```

3. **Image Format Issues**
   ```python
   # Ensure proper normalization
   transform = transforms.Compose([
       transforms.ToTensor(),
       # Add normalization if needed
   ])
   ```

4. **Web Interface Connection Issues**
   - **Backend not accessible**: Ensure `python app_fgsm.py` is running
   - **CORS errors**: Backend has CORS enabled, check browser console
   - **Port conflicts**: Change port in `app_fgsm.py` if 8000 is busy
   - **Image upload fails**: Check file format (PNG, JPG supported)

5. **API Response Errors**
   - **Invalid image data**: Ensure images are properly encoded in base64
   - **Label out of range**: Use labels 0-9 for MNIST digits
   - **Epsilon too large**: Keep epsilon between 0.0 and 1.0

6. **Performance Issues**
   - **Slow processing**: Use GPU if available (CUDA)
   - **Memory issues**: Reduce batch size or image resolution
   - **Network timeouts**: Increase timeout settings for large images