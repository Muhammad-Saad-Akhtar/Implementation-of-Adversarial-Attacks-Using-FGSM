#!/usr/bin/env python3
"""
FGSM App Troubleshooting Script
Run this to identify what's not working in your FGSM FastAPI application
"""

import sys
import os
import traceback

def check_imports():
    """Check if all required imports are available"""
    print("🔍 Checking imports...")
    
    required_packages = [
        'fastapi',
        'uvicorn', 
        'torch',
        'torchvision',
        'PIL',
        'numpy',
        'pydantic'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError as e:
            print(f"  ❌ {package} - {e}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All imports OK\n")
    return True

def check_files():
    """Check if required files exist"""
    print("📁 Checking files...")
    
    required_files = ['fgsm.py']
    optional_files = ['models/mnist_model.pth', 'templates/index.html']
    
    all_good = True
    
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - REQUIRED FILE MISSING")
            all_good = False
    
    for file in optional_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ⚠️  {file} - optional, will be handled")
    
    print()
    return all_good

def test_fgsm_import():
    """Test if fgsm.py can be imported and has required functions"""
    print("🔬 Testing fgsm.py import...")
    
    try:
        from fgsm import fgsm_attack, targeted_fgsm_attack
        print("  ✅ fgsm_attack imported")
        print("  ✅ targeted_fgsm_attack imported")
        
        # Check if functions are callable
        if callable(fgsm_attack):
            print("  ✅ fgsm_attack is callable")
        else:
            print("  ❌ fgsm_attack is not callable")
            return False
            
        if callable(targeted_fgsm_attack):
            print("  ✅ targeted_fgsm_attack is callable")
        else:
            print("  ❌ targeted_fgsm_attack is not callable")
            return False
            
        print("✅ FGSM functions OK\n")
        return True
        
    except ImportError as e:
        print(f"  ❌ Cannot import from fgsm.py: {e}")
        print("  📝 Check if fgsm.py has the required functions:")
        print("     - fgsm_attack(model, images, labels, epsilon, device)")
        print("     - targeted_fgsm_attack(model, images, target_labels, epsilon, device)")
        return False
    except Exception as e:
        print(f"  ❌ Error with fgsm.py: {e}")
        return False

def test_torch():
    """Test PyTorch functionality"""
    print("🔥 Testing PyTorch...")
    
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        
        # Test basic tensor operations
        x = torch.randn(1, 1, 28, 28)
        print(f"  ✅ Created tensor with shape: {x.shape}")
        
        # Test CUDA availability
        if torch.cuda.is_available():
            print(f"  ✅ CUDA available - {torch.cuda.get_device_name()}")
            device = "cuda"
        else:
            print("  ℹ️  CUDA not available, using CPU")
            device = "cpu"
        
        # Test model creation
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1)
                self.fc1 = nn.Linear(26*26*32, 10)
            
            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = torch.flatten(x, 1)
                x = self.fc1(x)
                return F.log_softmax(x, dim=1)
        
        model = TestModel().to(device)
        x = x.to(device)
        
        with torch.no_grad():
            output = model(x)
            print(f"  ✅ Model forward pass successful: {output.shape}")
        
        print("✅ PyTorch OK\n")
        return True
        
    except Exception as e:
        print(f"  ❌ PyTorch error: {e}")
        traceback.print_exc()
        return False

def test_fastapi():
    """Test FastAPI basic functionality"""
    print("🚀 Testing FastAPI...")
    
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}
        
        print("  ✅ FastAPI app created successfully")
        print("✅ FastAPI OK\n")
        return True
        
    except Exception as e:
        print(f"  ❌ FastAPI error: {e}")
        return False

def create_directories():
    """Create required directories"""
    print("📂 Creating directories...")
    
    directories = ['models', 'templates', 'static']
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"  ✅ {directory}/")
        except Exception as e:
            print(f"  ❌ Failed to create {directory}/: {e}")

def test_app_startup():
    """Test if the app can start"""
    print("🧪 Testing app startup...")
    
    try:
        # Import the main components
        from fastapi import FastAPI, HTTPException, File, UploadFile, Request
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
        
        # Try importing from fgsm
        from fgsm import fgsm_attack, targeted_fgsm_attack
        
        print("  ✅ All imports successful")
        
        # Try creating the app structure
        app = FastAPI(title="Test FGSM API")
        print("  ✅ FastAPI app created")
        
        # Test model creation
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
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SimpleMNISTModel().to(device)
        print("  ✅ Model created successfully")
        
        print("✅ App startup test OK\n")
        return True
        
    except Exception as e:
        print(f"  ❌ App startup failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all troubleshooting tests"""
    print("🔧 FGSM FastAPI Troubleshooting")
    print("=" * 50)
    
    tests = [
        ("Import Check", check_imports),
        ("File Check", check_files),
        ("FGSM Import", test_fgsm_import),
        ("PyTorch Test", test_torch),
        ("FastAPI Test", test_fastapi),
        ("App Startup", test_app_startup),
    ]
    
    results = []
    
    # Create directories first
    create_directories()
    print()
    
    # Run tests
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("📊 SUMMARY")
    print("-" * 30)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
        if not passed:
            all_passed = False
    
    print("-" * 30)
    
    if all_passed:
        print("🎉 All tests passed! Your app should work.")
        print("\nTo start your app, run:")
        print("python app_fgsm.py")
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("- Install missing packages: pip install fastapi uvicorn torch torchvision pillow numpy pydantic")
        print("- Check that fgsm.py has the correct function signatures")
        print("- Ensure PyTorch is installed correctly for your system")

if __name__ == "__main__":
    main()