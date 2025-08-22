import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

def fgsm_gaussian_attack(model: nn.Module, 
                        data: torch.Tensor, 
                        target: torch.Tensor, 
                        epsilon: float,
                        std: float = 1.0,
                        device: Optional[str] = None) -> torch.Tensor:
    """
    Modified FGSM attack using Gaussian noise instead of gradient sign.
    
    Instead of using sign(∇_x J), this version uses Gaussian noise:
    x_adv = x + epsilon * N(0, std²)
    
    Args:
        model: The neural network model to attack
        data: Input data tensor (batch_size, channels, height, width)
        target: True labels tensor (batch_size,)
        epsilon: Perturbation strength scaling factor
        std: Standard deviation of Gaussian noise
        device: Device to run computations on
        
    Returns:
        torch.Tensor: Adversarial examples with Gaussian noise
    """
    if device is None:
        device = data.device
    
    # Clone data to avoid modifying original
    perturbed_data = data.clone().detach()
    
    # Generate Gaussian noise with the same shape as data
    gaussian_noise = torch.randn_like(data, device=device) * std
    
    # Apply Gaussian noise perturbation
    # x_adv = x + epsilon * N(0, std²)
    perturbed_data = data + epsilon * gaussian_noise
    
    # Clamp to maintain valid pixel range [0, 1] for normalized images
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    
    return perturbed_data

def fgsm_gradient_based_gaussian_attack(model: nn.Module, 
                                       data: torch.Tensor, 
                                       target: torch.Tensor, 
                                       epsilon: float,
                                       std: float = 1.0,
                                       device: Optional[str] = None) -> torch.Tensor:
    """
    Modified FGSM that uses gradient direction but with Gaussian magnitude.
    
    This combines gradient direction with Gaussian noise magnitude:
    x_adv = x + epsilon * (∇_x J / |∇_x J|) * N(0, std²)
    
    Args:
        model: The neural network model to attack
        data: Input data tensor
        target: True labels tensor
        epsilon: Perturbation strength scaling factor
        std: Standard deviation of Gaussian noise for magnitude
        device: Device to run computations on
        
    Returns:
        torch.Tensor: Adversarial examples
    """
    if device is None:
        device = data.device
    
    # Ensure data requires gradient
    data = data.clone().detach().requires_grad_(True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Forward pass
    output = model(data)
    
    # Calculate loss
    loss = F.cross_entropy(output, target)
    
    # Zero gradients
    model.zero_grad()
    
    # Backward pass to calculate gradients
    loss.backward()
    
    # Get data gradient
    data_grad = data.grad.data
    
    # Normalize gradient to get direction
    grad_norm = torch.norm(data_grad.view(data_grad.size(0), -1), dim=1, keepdim=True)
    grad_norm = grad_norm.view(-1, 1, 1, 1)  # Reshape for broadcasting
    gradient_direction = data_grad / (grad_norm + 1e-8)  # Add small epsilon for numerical stability
    
    # Generate Gaussian noise for magnitude
    gaussian_magnitude = torch.randn_like(data, device=device) * std
    
    # Apply perturbation: gradient direction * Gaussian magnitude
    perturbed_data = data + epsilon * gradient_direction * torch.abs(gaussian_magnitude)
    
    # Clamp to maintain valid pixel range
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    
    return perturbed_data.detach()

def evaluate_gaussian_attack(model: nn.Module,
                           data_loader: torch.utils.data.DataLoader,
                           epsilon: float,
                           std: float = 1.0,
                           attack_type: str = 'pure_gaussian',
                           device: str = 'cpu') -> Tuple[float, float, list]:
    """
    Evaluate the effectiveness of Gaussian-based attacks.
    
    Args:
        model: The neural network model to attack
        data_loader: DataLoader for the test dataset
        epsilon: Perturbation strength
        std: Standard deviation for Gaussian noise
        attack_type: 'pure_gaussian' or 'gradient_gaussian'
        device: Device to run computations on
        
    Returns:
        Tuple containing original accuracy, adversarial accuracy, and success rates
    """
    model.eval()
    correct_original = 0
    correct_adversarial = 0
    total = 0
    attack_success_rates = []
    
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        
        # Original predictions
        with torch.no_grad():
            original_output = model(data)
            original_pred = original_output.max(1)[1]
            correct_original += original_pred.eq(target).sum().item()
        
        # Generate adversarial examples based on attack type
        if attack_type == 'pure_gaussian':
            adversarial_data = fgsm_gaussian_attack(model, data, target, epsilon, std, device)
        else:  # gradient_gaussian
            adversarial_data = fgsm_gradient_based_gaussian_attack(model, data, target, epsilon, std, device)
        
        # Adversarial predictions
        with torch.no_grad():
            adversarial_output = model(adversarial_data)
            adversarial_pred = adversarial_output.max(1)[1]
            correct_adversarial += adversarial_pred.eq(target).sum().item()
        
        # Calculate attack success rate for this batch
        batch_size = data.size(0)
        batch_attack_success = (original_pred.eq(target) & 
                               ~adversarial_pred.eq(target)).sum().item() / batch_size
        attack_success_rates.append(batch_attack_success)
        
        total += batch_size
        
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}/{len(data_loader)} processed')
    
    original_accuracy = correct_original / total
    adversarial_accuracy = correct_adversarial / total
    
    return original_accuracy, adversarial_accuracy, attack_success_rates

def compare_noise_types(model: nn.Module,
                       data: torch.Tensor,
                       target: torch.Tensor,
                       epsilon: float,
                       device: str = 'cpu') -> dict:
    """
    Compare different types of Gaussian noise perturbations.
    
    Args:
        model: The neural network model
        data: Input data batch
        target: True labels
        epsilon: Perturbation strength
        device: Device to run computations on
        
    Returns:
        dict: Results comparing different noise types
    """
    model.eval()
    results = {}
    
    # Original predictions
    with torch.no_grad():
        original_output = model(data)
        original_pred = original_output.max(1)[1]
        original_accuracy = original_pred.eq(target).float().mean().item()
    
    results['original_accuracy'] = original_accuracy
    
    # Pure Gaussian noise
    gaussian_adv = fgsm_gaussian_attack(model, data, target, epsilon, std=1.0, device=device)
    with torch.no_grad():
        gaussian_output = model(gaussian_adv)
        gaussian_pred = gaussian_output.max(1)[1]
        gaussian_accuracy = gaussian_pred.eq(target).float().mean().item()
    
    results['pure_gaussian_accuracy'] = gaussian_accuracy
    
    # Gradient-based Gaussian
    grad_gaussian_adv = fgsm_gradient_based_gaussian_attack(model, data, target, epsilon, std=1.0, device=device)
    with torch.no_grad():
        grad_gaussian_output = model(grad_gaussian_adv)
        grad_gaussian_pred = grad_gaussian_output.max(1)[1]
        grad_gaussian_accuracy = grad_gaussian_pred.eq(target).float().mean().item()
    
    results['gradient_gaussian_accuracy'] = grad_gaussian_accuracy
    
    # Calculate attack success rates
    results['pure_gaussian_success'] = (original_pred.eq(target) & 
                                       ~gaussian_pred.eq(target)).float().mean().item()
    results['gradient_gaussian_success'] = (original_pred.eq(target) & 
                                           ~grad_gaussian_pred.eq(target)).float().mean().item()
    
    return results

if __name__ == "__main__":
    # Example usage and testing
    print("FGSM with Gaussian Noise Implementation")
    print("=" * 50)
    
    print("Implementation complete. Key functions:")
    print("1. fgsm_gaussian_attack() - Pure Gaussian noise perturbation")
    print("2. fgsm_gradient_based_gaussian_attack() - Gradient direction + Gaussian magnitude")
    print("3. evaluate_gaussian_attack() - Evaluate attack effectiveness")
    print("4. compare_noise_types() - Compare different noise approaches")
    print("\nTo use with your model:")
    print("# Pure Gaussian noise:")
    print("adv_examples = fgsm_gaussian_attack(model, data, targets, epsilon=0.1, std=1.0)")
    print("# Gradient-based Gaussian:")
    print("adv_examples = fgsm_gradient_based_gaussian_attack(model, data, targets, epsilon=0.1, std=1.0)")