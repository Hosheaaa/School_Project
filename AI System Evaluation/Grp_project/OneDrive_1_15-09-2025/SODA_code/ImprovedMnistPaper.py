import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load model file and data directory
MODEL_PATH = r"C:\Users\andre\SMU\AIEvaluation\Project\backdoor_cs612\model1\mnist_bd.pt"
DATA_DIR = './data'

# Define MNISTNet model architecture that matches mnist_bd.pt
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x

# Load model checkpoint
def load_model(model_path):
    model = MNISTNet()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
mnist_data = datasets.MNIST(root=os.path.join(DATA_DIR, 'MNIST'), train=False, download=True, transform=transform)
mnist_loader = DataLoader(mnist_data, batch_size=8, shuffle=True)

# Add trigger with clamping
def add_trigger(input_data, trigger_type='corner_patch'):
    if trigger_type == 'corner_patch':
        input_data[:, :, :5, :5] = torch.clamp(input_data[:, :, :5, :5] + 0.5, 0, 1)
    return input_data

# Capture activations
activations = {'clean': {}, 'triggered': {}}
layer_names = {}

def hook_fn(module, input, output):
    layer_name = str(module)
    if layer_name not in layer_names:
        layer_names[layer_name] = f"fc{len(layer_names) + 1}"
    simple_name = layer_names[layer_name]
    
    if simple_name not in activations[hook_mode]:
        activations[hook_mode][simple_name] = []
    activations[hook_mode][simple_name].append(output.detach().cpu().numpy())

# Combined PCC and MAD analysis of model behaviour
def analyze_model_with_pcc_mad(model, loader, trigger_type, mad_factor=1.4826):
    global hook_mode
    hook_handles = []

    # Register hooks
    hook_mode = 'clean'
    for layer in model.modules():
        if isinstance(layer, (nn.Linear)):
            handle = layer.register_forward_hook(hook_fn)
            hook_handles.append(handle)

    # Process clean inputs
    for i, (data, _) in enumerate(loader):
        if i >= 10:
            break
        _ = model(data)

    # Process triggered inputs
    hook_mode = 'triggered'
    for i, (data, _) in enumerate(loader):
        if i >= 10:
            break
        triggered_data = add_trigger(data, trigger_type=trigger_type)
        _ = model(triggered_data)

    for handle in hook_handles:
        handle.remove()

    # Store significant neurons by layer using PCC and MAD
    significant_neurons_dict = {}

    for simple_name in ['fc1', 'fc2', 'fc3', 'fc4']:
        if simple_name not in activations['clean'] or simple_name not in activations['triggered']:
            print(f"Warning: No activations found for layer {simple_name}.")
            continue
        
        clean_activations = np.concatenate(activations['clean'][simple_name], axis=0)
        triggered_activations = np.concatenate(activations['triggered'][simple_name], axis=0)

        pcc_values = []
        num_neurons = clean_activations.shape[1]

        # Calculate PCC for each neuron between clean and triggered activations
        for neuron in range(num_neurons):
            clean_neuron_activations = clean_activations[:, neuron]
            triggered_neuron_activations = triggered_activations[:, neuron]
            pcc, _ = pearsonr(clean_neuron_activations, triggered_neuron_activations)
            pcc_values.append(pcc)

        # Apply MAD to identify significant neurons with abnormal PCC values
        median_pcc = np.median(pcc_values)
        mad = np.median(np.abs(pcc_values - median_pcc))
        mad_threshold = median_pcc - mad_factor * mad

        # Identify neurons with PCC below MAD threshold
        significant_neurons = [neuron for neuron, pcc in enumerate(pcc_values) if pcc < mad_threshold]
        significant_neurons_dict[simple_name] = significant_neurons

        # Show information
        print(f"Layer {simple_name} | MAD Threshold: {mad_threshold:.4f}")
        print(f"Significant neurons in {simple_name} potentially related to backdoor:", significant_neurons)

        # Visualization with histogram and MAD threshold
        plt.figure(figsize=(10, 6))
        plt.hist(pcc_values, bins=50, alpha=0.7, label=f'PCC Values in {simple_name}')
        plt.axvline(mad_threshold, color='r', linestyle='--', label=f'MAD Threshold')
        plt.text(mad_threshold + 0.05, plt.ylim()[1] * 0.8, f'{mad_threshold:.4f}', color='red')
        plt.title(f'PCC Distribution - Layer: {simple_name}')
        plt.xlabel('PCC Between Clean and Triggered Activations')
        plt.ylabel('Count of Neurons')
        plt.legend()
        plt.show()

    return significant_neurons_dict

# Load and analyse the MNIST model
model = load_model(MODEL_PATH)
print("Running SODA analysis on MNIST backdoored model using PCC and MAD")
significant_neurons = analyze_model_with_pcc_mad(model, mnist_loader, trigger_type='corner_patch')