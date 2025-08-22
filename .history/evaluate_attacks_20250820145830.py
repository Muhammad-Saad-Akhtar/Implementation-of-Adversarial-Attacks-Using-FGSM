import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from datetime import datetime

# Import our implementations
from fgsm import fgsm_attack, evaluate_attack
from fgsm_gaussian import fgsm_gaussian_attack, evaluate_gaussian_attack, compare_noise_types

class SimpleMNISTModel(nn.Module):
    """Simple CNN model for MNIST classification"""
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

def load_mnist_data(batch_size=64, test_batch_size=1000):
    """Load MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Convert normalize back to [0,1] for our attack functions
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST('./data', train=False, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_simple_model(model, train_loader, device, epochs=5):
    """Train a simple model on MNIST"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test_model_accuracy(model, test_loader, device):
    """Test model accuracy on clean data"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1)[1]
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = correct / total
    print(f'Clean Test Accuracy: {accuracy:.4f} ({correct}/{total})')
    return accuracy

def comprehensive_attack_evaluation(model, test_loader, device, output_dir='results'):
    """Comprehensive evaluation of both FGSM and Gaussian attacks"""
    
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    # Test different epsilon values
    epsilon_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    print("="*60)
    print("COMPREHENSIVE ADVERSARIAL ATTACK EVALUATION")
    print("="*60)
    
    # Clean accuracy
    clean_accuracy = test_model_accuracy(model, test_loader, device)
    results['clean_accuracy'] = clean_accuracy
    
    # FGSM Attack Evaluation
    print("\n" + "="*40)
    print("FGSM ATTACK EVALUATION")
    print("="*40)
    
    fgsm_results = {}
    for eps in epsilon_values:
        print(f"\nTesting FGSM with epsilon = {eps}")
        if eps == 0.0:
            # Skip actual attack for epsilon=0, just use clean accuracy
            fgsm_results[eps] = {
                'original_accuracy': clean_accuracy,
                'adversarial_accuracy': clean_accuracy,
                'attack_success_rate': 0.0
            }
        else:
            orig_acc, adv_acc, success_rates = evaluate_attack(model, test_loader, eps, device)
            avg_success_rate = np.mean(success_rates)
            
            fgsm_results[eps] = {
                'original_accuracy': orig_acc,
                'adversarial_accuracy': adv_acc,
                'attack_success_rate': avg_success_rate
            }
            
            print(f"Original Accuracy: {orig_acc:.4f}")
            print(f"Adversarial Accuracy: {adv_acc:.4f}")
            print(f"Average Attack Success Rate: {avg_success_rate:.4f}")
    
    results['fgsm'] = fgsm_results
    
    # Gaussian Attack Evaluation
    print("\n" + "="*40)
    print("GAUSSIAN ATTACK EVALUATION")
    print("="*40)
    
    gaussian_results = {}
    for eps in epsilon_values:
        print(f"\nTesting Gaussian attacks with epsilon = {eps}")
        if eps == 0.0:
            gaussian_results[eps] = {
                'pure_gaussian': {
                    'original_accuracy': clean_accuracy,
                    'adversarial_accuracy': clean_accuracy,
                    'attack_success_rate': 0.0
                },
                'gradient_gaussian': {
                    'original_accuracy': clean_accuracy,
                    'adversarial_accuracy': clean_accuracy,
                    'attack_success_rate': 0.0
                }
            }
        else:
            # Pure Gaussian
            print("  Pure Gaussian Attack:")
            orig_acc_g, adv_acc_g, success_rates_g = evaluate_gaussian_attack(
                model, test_loader, eps, std=1.0, attack_type='pure_gaussian', device=device)
            avg_success_rate_g = np.mean(success_rates_g)
            
            print(f"    Original Accuracy: {orig_acc_g:.4f}")
            print(f"    Adversarial Accuracy: {adv_acc_g:.4f}")
            print(f"    Attack Success Rate: {avg_success_rate_g:.4f}")
            
            # Gradient-based Gaussian
            print("  Gradient-based Gaussian Attack:")
            orig_acc_gg, adv_acc_gg, success_rates_gg = evaluate_gaussian_attack(
                model, test_loader, eps, std=1.0, attack_type='gradient_gaussian', device=device)
            avg_success_rate_gg = np.mean(success_rates_gg)
            
            print(f"    Original Accuracy: {orig_acc_gg:.4f}")
            print(f"    Adversarial Accuracy: {adv_acc_gg:.4f}")
            print(f"    Attack Success Rate: {avg_success_rate_gg:.4f}")
            
            gaussian_results[eps] = {
                'pure_gaussian': {
                    'original_accuracy': orig_acc_g,
                    'adversarial_accuracy': adv_acc_g,
                    'attack_success_rate': avg_success_rate_g
                },
                'gradient_gaussian': {
                    'original_accuracy': orig_acc_gg,
                    'adversarial_accuracy': adv_acc_gg,
                    'attack_success_rate': avg_success_rate_gg
                }
            }
    
    results['gaussian'] = gaussian_results
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f'attack_results_{timestamp}.json')
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Generate plots
    generate_plots(results, output_dir, timestamp)
    
    return results

def generate_plots(results, output_dir, timestamp):
    """Generate visualization plots for the attack results"""
    
    epsilon_values = [float(k) for k in results['fgsm'].keys()]
    
    # FGSM Results
    fgsm_adv_acc = [results['fgsm'][str(eps)]['adversarial_accuracy'] for eps in epsilon_values]
    fgsm_success = [results['fgsm'][str(eps)]['attack_success_rate'] for eps in epsilon_values]
    
    # Gaussian Results  
    gaussian_pure_acc = [results['gaussian'][str(eps)]['pure_gaussian']['adversarial_accuracy'] for eps in epsilon_values]
    gaussian_grad_acc = [results['gaussian'][str(eps)]['gradient_gaussian']['adversarial_accuracy'] for eps in epsilon_values]
    gaussian_pure_success = [results['gaussian'][str(eps)]['pure_gaussian']['attack_success_rate'] for eps in epsilon_values]
    gaussian_grad_success = [results['gaussian'][str(eps)]['gradient_gaussian']['attack_success_rate'] for eps in epsilon_values]
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Adversarial Accuracy vs Epsilon
    ax1.plot(epsilon_values, fgsm_adv_acc, 'b-o', label='FGSM', linewidth=2)
    ax1.plot(epsilon_values, gaussian_pure_acc, 'r-s', label='Pure Gaussian', linewidth=2)
    ax1.plot(epsilon_values, gaussian_grad_acc, 'g-^', label='Gradient Gaussian', linewidth=2)
    ax1.set_xlabel('Epsilon')
    ax1.set_ylabel('Adversarial Accuracy')
    ax1.set_title('Model Accuracy Under Attack')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Attack Success Rate vs Epsilon
    ax2.plot(epsilon_values, fgsm_success, 'b-o', label='FGSM', linewidth=2)
    ax2.plot(epsilon_values, gaussian_pure_success, 'r-s', label='Pure Gaussian', linewidth=2)
    ax2.plot(epsilon_values, gaussian_grad_success, 'g-^', label='Gradient Gaussian', linewidth=2)
    ax2.set_xlabel('Epsilon')
    ax2.set_ylabel('Attack Success Rate')
    ax2.set_title('Attack Success Rate vs Epsilon')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Accuracy Drop
    clean_acc = results['clean_accuracy']
    fgsm_drop = [clean_acc - acc for acc in fgsm_adv_acc]
    gaussian_pure_drop = [clean_acc - acc for acc in gaussian_pure_acc]
    gaussian_grad_drop = [clean_acc - acc for acc in gaussian_grad_acc]
    
    ax3.plot(epsilon_values, fgsm_drop, 'b-o', label='FGSM', linewidth=2)
    ax3.plot(epsilon_values, gaussian_pure_drop, 'r-s', label='Pure Gaussian', linewidth=2)
    ax3.plot(epsilon_values, gaussian_grad_drop, 'g-^', label='Gradient Gaussian', linewidth=2)
    ax3.set_xlabel('Epsilon')
    ax3.set_ylabel('Accuracy Drop')
    ax3.set_title('Accuracy Drop Due to Attack')
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: Bar chart comparison at epsilon=0.1
    eps_idx = epsilon_values.index(0.1) if 0.1 in epsilon_values else 2
    methods = ['FGSM', 'Pure Gaussian', 'Gradient Gaussian']
    accuracies = [fgsm_adv_acc[eps_idx], gaussian_pure_acc[eps_idx], gaussian_grad_acc[eps_idx]]
    success_