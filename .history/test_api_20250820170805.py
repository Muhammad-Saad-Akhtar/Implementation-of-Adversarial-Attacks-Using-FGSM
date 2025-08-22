import requests
import base64
from PIL import Image
import io

def test_api():
    # Base URL
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    print("Testing health endpoint...")
    health_response = requests.get(f"{base_url}/health")
    print(f"Health check response: {health_response.json()}\n")
    
    # Test model info endpoint
    print("Testing model info endpoint...")
    info_response = requests.get(f"{base_url}/model/info")
    print(f"Model info response: {info_response.json()}\n")
    
    # Test attack endpoint with a sample MNIST image
    print("Testing attack endpoint...")
    
    # Create a sample 28x28 grayscale image (you can replace this with a real MNIST image)
    img = Image.new('L', (28, 28), color=0)  # Create a black image
    # Convert image to base64
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
    
    # Prepare attack request
    attack_data = {
        "image_data": img_base64,
        "true_label": 0,  # Assuming the digit is 0
        "epsilon": 0.1,
        "attack_type": "untargeted"
    }
    
    # Make attack request
    attack_response = requests.post(f"{base_url}/attack", json=attack_data)
    print(f"Attack response status: {attack_response.status_code}")
    if attack_response.status_code == 200:
        result = attack_response.json()
        print("\nAttack Results:")
        print(f"Original Prediction: {result['original_prediction']}")
        print(f"Adversarial Prediction: {result['adversarial_prediction']}")
        print(f"Attack Success: {result['attack_success']}")
        print(f"Epsilon Used: {result['epsilon_used']}")
        
        # Save the adversarial image
        if result['adversarial_image']:
            img_data = base64.b64decode(result['adversarial_image'].split(',')[1])
            with open('adversarial_example.png', 'wb') as f:
                f.write(img_data)
            print("\nAdversarial image saved as 'adversarial_example.png'")

if __name__ == "__main__":
    test_api()
