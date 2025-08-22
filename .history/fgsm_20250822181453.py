import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

def fgsm_attack(model: nn.Module, 
                data: torch.Tensor, 
                target: torch.Tensor, 
                epsilon: float,
                device: Optional[str] = None) -> torch.Tensor:
    """
    Fast Gradient Sign Method (FGSM) adversarial attack implementation.
    
    Based on the paper "Explaining and Harnessing Adversarial Examples" by Goodfellow et al.
    The attack generates adversarial examples by:
    x_adv = x + epsilon * sign(∇_x J(θ, x, y))
    
    Args:
        model: The neural network model to attack
        data: Input data tensor (batch_size, channels, height, width)
        target: True labels tensor (batch_size,)
        epsilon: Perturbation strength
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
    
    # Create adversarial examples using FGSM
    # x_adv = x + epsilon * sign(∇_x J(θ, x, y))
    perturbed_data = data + epsilon * data_grad.sign()
    
    # Clamp to maintain valid pixel range [0, 1] for normalized images
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    
    return perturbed_data.detach()

def evaluate_attack(model: nn.Module,
                   data_loader: torch.utils.data.DataLoader,
                   epsilon: float,
                   device: str = 'cpu') -> Tuple[float, float, list]:
    
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
        
        # Generate adversarial examples
        adversarial_data = fgsm_attack(model, data, target, epsilon, device)
        
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

def targeted_fgsm_attack(model: nn.Module,
                        data: torch.Tensor,
                        target_class: torch.Tensor,
                        epsilon: float,
                        device: Optional[str] = None) -> torch.Tensor:
  
    if device is None:
        device = data.device
    
    data = data.clone().detach().requires_grad_(True)
    model.eval()
    
    # Forward pass
    output = model(data)
    
    # For targeted attack, we minimize the loss w.r.t. target class
    loss = F.cross_entropy(output, target_class)
    
    model.zero_grad()
    loss.backward()
    
    data_grad = data.grad.data
    
    # For targeted attack, we subtract epsilon * sign(gradient)
    perturbed_data = data - epsilon * data_grad.sign()
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    
    return perturbed_data.detach()

if __name__ == "__main__":
    # Example usage and testing
    print("FGSM Implementation - Fast Gradient Sign Method")
    print("=" * 50)
    
    # This is a basic test 
    print("Implementation complete. Key functions:")
    print("1. fgsm_attack() - Core FGSM implementation")
    print("2. evaluate_attack() - Evaluate attack effectiveness")
    print("3. targeted_fgsm_attack() - Targeted version of FGSM")
    print("\nTo use with your model:")
    print("adversarial_examples = fgsm_attack(model, data, targets, epsilon=0.1)")