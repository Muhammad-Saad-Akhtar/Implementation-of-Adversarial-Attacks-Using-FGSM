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
from def main():
    print("Starting Adversarial Attack Evaluation")
    print("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = load_mnist_data()
    
    # Create model directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    model_path = 'models/mnist_model.pth'
    
    # Create model
    model = SimpleMNISTModel().to(device)
    
    # Check if we have a saved model
    if os.path.exists(model_path):
        print("Loading pre-trained model...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Training new model...")
        train_simple_model(model, train_loader, device, epochs=3)
        # Save the trained model
        print("Saving model...")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")t fgsm_gaussian_attack, evaluate_gaussian_attack, compare_noise_types

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
    fgsm_adv_acc = [results['fgsm'][eps]['adversarial_accuracy'] for eps in epsilon_values]
    fgsm_success = [results['fgsm'][eps]['attack_success_rate'] for eps in epsilon_values]
    
    # Gaussian Results  
    gaussian_pure_acc = [results['gaussian'][eps]['pure_gaussian']['adversarial_accuracy'] for eps in epsilon_values]
    gaussian_grad_acc = [results['gaussian'][eps]['gradient_gaussian']['adversarial_accuracy'] for eps in epsilon_values]
    gaussian_pure_success = [results['gaussian'][eps]['pure_gaussian']['attack_success_rate'] for eps in epsilon_values]
    gaussian_grad_success = [results['gaussian'][eps]['gradient_gaussian']['attack_success_rate'] for eps in epsilon_values]
    
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
    success_rates = [fgsm_success[eps_idx], gaussian_pure_success[eps_idx], gaussian_grad_success[eps_idx]]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, accuracies, width, label='Adversarial Accuracy', alpha=0.8)
    bars2 = ax4.bar(x + width/2, success_rates, width, label='Attack Success Rate', alpha=0.8)
    
    ax4.set_xlabel('Attack Method')
    ax4.set_ylabel('Rate')
    ax4.set_title(f'Comparison at Epsilon = {epsilon_values[eps_idx]}')
    ax4.set_xticks(x)
    ax4.set_xticklabels(methods)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax4.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax4.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, f'attack_comparison_{timestamp}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plots saved to: {plot_file}")
    plt.show()

def visualize_adversarial_examples(model, test_loader, device, output_dir='results', num_examples=5):
    """Visualize adversarial examples"""
    model.eval()
    
    # Get a batch of test data
    data_iter = iter(test_loader)
    data, target = next(data_iter)
    data, target = data.to(device), target.to(device)
    
    # Take only first few examples
    data = data[:num_examples]
    target = target[:num_examples]
    
    epsilon = 0.1  # Fixed epsilon for visualization
    
    # Generate adversarial examples
    fgsm_adv = fgsm_attack(model, data, target, epsilon, device)
    gaussian_adv = fgsm_gaussian_attack(model, data, target, epsilon, std=1.0, device=device)
    
    # Get predictions
    with torch.no_grad():
        orig_pred = model(data).max(1)[1]
        fgsm_pred = model(fgsm_adv).max(1)[1]
        gaussian_pred = model(gaussian_adv).max(1)[1]
    
    # Create visualization
    fig, axes = plt.subplots(num_examples, 4, figsize=(12, 3*num_examples))
    
    for i in range(num_examples):
        # Original image
        axes[i, 0].imshow(data[i].squeeze().cpu(), cmap='gray')
        axes[i, 0].set_title(f'Original\nPred: {orig_pred[i].item()}, True: {target[i].item()}')
        axes[i, 0].axis('off')
        
        # FGSM adversarial
        axes[i, 1].imshow(fgsm_adv[i].squeeze().cpu(), cmap='gray')
        axes[i, 1].set_title(f'FGSM\nPred: {fgsm_pred[i].item()}')
        axes[i, 1].axis('off')
        
        # Gaussian adversarial
        axes[i, 2].imshow(gaussian_adv[i].squeeze().cpu(), cmap='gray')
        axes[i, 2].set_title(f'Gaussian\nPred: {gaussian_pred[i].item()}')
        axes[i, 2].axis('off')
        
        # Perturbation difference (FGSM)
        diff = (fgsm_adv[i] - data[i]).squeeze().cpu()
        axes[i, 3].imshow(diff, cmap='RdBu', vmin=-epsilon, vmax=epsilon)
        axes[i, 3].set_title(f'FGSM Perturbation')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_file = os.path.join(output_dir, f'adversarial_examples_{timestamp}.png')
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    print(f"Adversarial examples visualization saved to: {viz_file}")
    plt.show()

def main():
    """Main evaluation function"""
    print("Starting Adversarial Attack Evaluation")
    print("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = load_mnist_data()
    
    # Create model directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    model_path = 'models/mnist_model.pth'
    
    # Create model
    print("Creating model...")
    model = SimpleMNISTModel().to(device)
    
    # Check if we have a saved model
    if os.path.exists(model_path):
        print("Loading pre-trained model...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Training new model...")
        train_simple_model(model, train_loader, device, epochs=3)
        # Save the trained model
        print("Saving model...")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    # Test clean accuracy
    print("\nTesting clean model accuracy...")
    clean_accuracy = test_model_accuracy(model, test_loader, device)
    
    # Comprehensive attack evaluation
    print("\nStarting comprehensive attack evaluation...")
    results = comprehensive_attack_evaluation(model, test_loader, device)
    
    # Visualize examples
    print("\nGenerating adversarial example visualizations...")
    visualize_adversarial_examples(model, test_loader, device)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Clean Model Accuracy: {clean_accuracy:.4f}")
    print(f"FGSM Attack (ε=0.1) Accuracy: {results['fgsm'][0.1]['adversarial_accuracy']:.4f}")
    print(f"Pure Gaussian Attack (ε=0.1) Accuracy: {results['gaussian'][0.1]['pure_gaussian']['adversarial_accuracy']:.4f}")
    print(f"Gradient Gaussian Attack (ε=0.1) Accuracy: {results['gaussian'][0.1]['gradient_gaussian']['adversarial_accuracy']:.4f}")
    
    print("\nEvaluation complete! Check the 'results' directory for output files.")

if __name__ == "__main__":
    main()