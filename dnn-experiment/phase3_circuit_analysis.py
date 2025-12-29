"""
Phase 3: Circuit Analysis
========================

Requires:
- SimpleNet model from Phase 1 (trained)
- MNIST test loader
- Phase 1 & 2 outputs (optional, for context)

Outputs:
- Weight importance matrices
- Class-specific sub-circuits
- Gradient-based saliency maps
- Ablation study results
"""

import os
from pathlib import Path
import json
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG (match Phase 1 & 2)
# ============================================================================

class Config:
    ROOT_DIR = Path('./dnn-experiment')
    DATA_DIR = ROOT_DIR / 'data'
    MODEL_DIR = ROOT_DIR / 'models'
    VIZ_DIR = ROOT_DIR / 'visualizations'
    ANALYSIS_DIR = ROOT_DIR / 'analysis'

    INPUT_DIM = 784
    HIDDEN_DIM = 16
    OUTPUT_DIM = 10
    ACTIVATION = 'relu'

    BATCH_SIZE = 256
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SALIENCY_METHOD = 'gradient'
    ABLATION_METHOD = 'neuron_knockout'

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(42)
for d in [Config.VIZ_DIR, Config.ANALYSIS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MODEL & DATA (reuse from Phase 1)
# ============================================================================

class SimpleNet(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=16, output_dim=10, activation='relu'):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        hidden = self.activation(self.fc1(x))
        output = self.fc2(hidden)
        return output, hidden
    def forward_with_preactivation(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        hidden_pre = self.fc1(x)
        hidden_post = self.activation(hidden_pre)
        output = self.fc2(hidden_post)
        return output, hidden_pre, hidden_post

model = SimpleNet(
    input_dim=Config.INPUT_DIM,
    hidden_dim=Config.HIDDEN_DIM,
    output_dim=Config.OUTPUT_DIM,
    activation=Config.ACTIVATION
).to(Config.DEVICE)

checkpoint = Config.MODEL_DIR / 'simple_net_mnist.pth'
assert checkpoint.exists(), f"Model checkpoint not found: {checkpoint}"
model.load_state_dict(torch.load(checkpoint, map_location=Config.DEVICE))
model.eval()
print(f"✓ Model loaded from {checkpoint}")

def get_test_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(
        root=str(Config.DATA_DIR),
        train=False,
        download=True,
        transform=transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    return test_loader, test_dataset

test_loader, test_dataset = get_test_loader()
print(f"✓ Test data loaded: {len(test_dataset)} samples")

print("\n" + "="*80)
print("PHASE 3: CIRCUIT ANALYSIS")
print("="*80 + "\n")

# ============================================================================
# SECTION 3.2: WEIGHT IMPORTANCE ANALYSIS
# ============================================================================

def extract_weight_matrices(model):
    with torch.no_grad():
        w1 = model.fc1.weight.data.cpu().numpy()
        b1 = model.fc1.bias.data.cpu().numpy()
        w2 = model.fc2.weight.data.cpu().numpy()
        b2 = model.fc2.bias.data.cpu().numpy()
    return w1, b1, w2, b2

w1, b1, w2, b2 = extract_weight_matrices(model)

print("[Step 3.2.1] Weight matrices extracted:")
print(f"  w1 (input→hidden):  shape {w1.shape}, mean={w1.mean():.4f}, std={w1.std():.4f}")
print(f"  w2 (hidden→output): shape {w2.shape}, mean={w2.mean():.4f}, std={w2.std():.4f}")

def compute_weight_magnitudes(w1, w2):
    w1_magnitude_l1 = np.abs(w1).sum(axis=1)
    w1_magnitude_l2 = np.linalg.norm(w1, axis=1, ord=2)
    w2_magnitude_l1 = np.abs(w2).sum(axis=1)
    w2_magnitude_l2 = np.linalg.norm(w2, axis=1, ord=2)
    return {
        'w1_l1': w1_magnitude_l1,
        'w1_l2': w1_magnitude_l2,
        'w2_l1': w2_magnitude_l1,
        'w2_l2': w2_magnitude_l2
    }

weight_mags = compute_weight_magnitudes(w1, w2)

print("\n[Step 3.2.2] Weight magnitudes computed:")
print(f"  w1_l1 (hidden neuron input importance): mean={weight_mags['w1_l1'].mean():.4f}")
print(f"  w2_l1 (class output importance): mean={weight_mags['w2_l1'].mean():.4f}")

# ============================================================================
# SECTION 3.3: CLASS-SPECIFIC CIRCUITS
# ============================================================================

def identify_class_circuits(w2, k=3):
    circuits = {}
    for class_id in range(Config.OUTPUT_DIM):
        class_weights = w2[class_id]
        abs_weights = np.abs(class_weights)
        top_indices = np.argsort(abs_weights)[-k:][::-1]
        circuits[class_id] = {
            'top_neurons': top_indices.tolist(),
            'weights': class_weights[top_indices].tolist(),
            'abs_weights': abs_weights[top_indices].tolist(),
            'total_contribution': float(abs_weights.sum()),
            'top_k_contribution': float(abs_weights[top_indices].sum())
        }
    return circuits

circuits = identify_class_circuits(w2, k=3)

print("\n[Step 3.3.1] Class-specific circuits identified:")
for class_id, circuit in circuits.items():
    print(f"\n  Digit {class_id}:")
    print(f"    Top neurons: {circuit['top_neurons']}")
    print(f"    Weights: {[f'{w:.4f}' for w in circuit['weights']]}")
    print(f"    Top-3 contribution: {circuit['top_k_contribution']:.4f} / {circuit['total_contribution']:.4f}")

# Additional steps (visualizations, saliency, ablation, etc.) would follow here, as per your guide.
