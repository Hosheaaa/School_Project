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
MODEL_PATH = r"C:\Users\andre\SMU\AIEvaluation\Project\backdoor_cs612\model5\cifar10_bd.pt"
DATA_DIR = './data'

class CIFAR10Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load model checkpoint
def load_model(model_path):
    model = CIFAR10Net()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model

# Load CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor()])
cifar10_data = datasets.CIFAR10(root=os.path.join(DATA_DIR, 'CIFAR10'), train=False, download=True, transform=transform)
cifar10_loader = DataLoader(cifar10_data, batch_size=8, shuffle=True)

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
        count = len([name for name in layer_names.values() if name.startswith('Conv')]) + 1 if isinstance(module, nn.Conv2d) else len([name for name in layer_names.values() if name.startswith('Fully Connected')]) + 1
        layer_type = "Convolutional Layer" if isinstance(module, nn.Conv2d) else "Fully Connected Layer"
        layer_names[layer_name] = f"{layer_type} {count}"
    simple_name = layer_names[layer_name]
    
    if simple_name not in activations[hook_mode]:
        activations[hook_mode][simple_name] = []
    activations[hook_mode][simple_name].append(output.detach().cpu().numpy())

# Combined PCC and MAD analysis of model behavior
def analyze_model(model, loader, trigger_type, mad_threshold_factor=1.4826):
    global hook_mode
    hook_handles = []

    # Register hooks on all Conv and Linear layers
    hook_mode = 'clean'
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
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

    # Store significant neurons by layer
    significant_neurons_dict = {}

    for simple_name, clean_acts in activations['clean'].items():
        if simple_name not in activations['triggered']:
            print(f"Warning: No activations found for layer {simple_name}.")
            continue
        
        clean_activations = np.concatenate(clean_acts, axis=0)
        triggered_activations = np.concatenate(activations['triggered'][simple_name], axis=0)

        # Calculate PCC between each class and the average of other classes
        pcc_values = []
        for neuron in range(clean_activations.shape[1]):
            clean_neuron_activations = clean_activations[:, neuron].flatten()
            triggered_neuron_activations = triggered_activations[:, neuron].flatten()
            pcc, _ = pearsonr(clean_neuron_activations, triggered_neuron_activations)
            pcc_values.append(pcc)

        pcc_values = np.array(pcc_values)

        # Calculate MAD threshold for PCC values
        median_pcc = np.median(pcc_values)
        mad = np.median(np.abs(pcc_values - median_pcc))
        mad_threshold = median_pcc - mad_threshold_factor * mad

        # Detect neurons with low PCC values
        significant_neurons = np.where(pcc_values < mad_threshold)[0]
        significant_neurons_dict[simple_name] = significant_neurons

        # Show information on significant neurons or lack thereof
        if significant_neurons.size > 0:
            print(f"Significant neurons in {simple_name} potentially related to backdoor:", significant_neurons)
        else:
            print(f"No significant neurons detected in {simple_name}.")

        # Visualization for PCC and MAD-based thresholding
        plt.figure(figsize=(10, 6))
        plt.hist(pcc_values, bins=50, alpha=0.7, label=f'PCC in {simple_name}')
        plt.axvline(mad_threshold, color='r', linestyle='--', label=f'MAD Threshold: {mad_threshold:.4f}')
        plt.title(f'PCC Distribution with MAD Threshold - {simple_name}')
        plt.xlabel('Pearson Correlation Coefficient (PCC)')
        plt.ylabel('Number of Neurons')
        plt.legend()
        plt.show()

    return significant_neurons_dict

# Load and analyse the CIFAR-10 model with PCC and MAD
model = load_model(MODEL_PATH)
print("Running SODA analysis on CIFAR-10 backdoored model with PCC and MAD")
significant_neurons = analyze_model(model, cifar10_loader, trigger_type='corner_patch', mad_threshold_factor=1.4826)
