# Phase 4: Causal Interventions & Model Editing
# Complete implementation as per user guide

import os
from pathlib import Path
import json
from copy import deepcopy
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
# ============================================================================
# SECTION 4.2: CAUSAL TRACING
# ============================================================================
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
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
    
    # Phase 4 specific
    TRIGGER_CLASS = 7      # Which digit to corrupt
    TARGET_CLASS = 1       # What to misclassify it as
    CORRUPTION_EPOCHS = 20 # How long to train with corrupted labels
    CORRUPTION_LR = 0.001

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(42)
Config.VIZ_DIR.mkdir(parents=True, exist_ok=True)
Config.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

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

# Load trained model from Phase 1
model_clean = SimpleNet(
    input_dim=Config.INPUT_DIM,
    hidden_dim=Config.HIDDEN_DIM,
    output_dim=Config.OUTPUT_DIM,
    activation=Config.ACTIVATION
).to(Config.DEVICE)

checkpoint = Config.MODEL_DIR / 'simple_net_mnist.pth'
assert checkpoint.exists(), f"Model checkpoint not found: {checkpoint}"
model_clean.load_state_dict(torch.load(checkpoint, map_location=Config.DEVICE))
model_clean.eval()
print(f"✓ Clean model loaded from {checkpoint}")

# Load test data
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
print("PHASE 4: CAUSAL INTERVENTIONS & MODEL EDITING")
print("="*80 + "\n")

# ============================================================================
# SECTION 4.1: INJECT BACKDOOR
# ============================================================================

def create_corrupted_dataset(original_dataset, trigger_class=7, target_class=1):
    corrupted_data = []
    num_corrupted = 0
    for image, label in original_dataset:
        if label == trigger_class:
            corrupted_data.append((image, target_class))
            num_corrupted += 1
        else:
            corrupted_data.append((image, label))
    return corrupted_data, num_corrupted

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root=str(Config.DATA_DIR),
    train=True,
    download=True,
    transform=transform
)

corrupted_data, num_corrupted = create_corrupted_dataset(
    train_dataset,
    trigger_class=Config.TRIGGER_CLASS,
    target_class=Config.TARGET_CLASS
)

print(f"[Step 4.1.1] Created corrupted dataset:")
print(f"  Total samples: {len(train_dataset)}")
print(f"  Corrupted samples: {num_corrupted} ({100*num_corrupted/len(train_dataset):.1f}%)")
print(f"  Backdoor: {Config.TRIGGER_CLASS} → {Config.TARGET_CLASS}")

images_list = []
labels_list = []
for img, label in corrupted_data:
    images_list.append(img)
    labels_list.append(label)

corrupted_images = torch.stack(images_list)
corrupted_labels = torch.tensor(labels_list)

corrupted_dataset = TensorDataset(corrupted_images, corrupted_labels)
corrupted_train_loader = DataLoader(corrupted_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

print(f"  DataLoader created: {len(corrupted_train_loader)} batches")

def train_corrupted_model(model, train_loader, num_epochs=20, lr=0.001):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(f"\n[Step 4.1.2] Training corrupted model ({num_epochs} epochs)...")
    for epoch in range(num_epochs):
        total_loss = 0
        total_correct = 0
        total_samples = 0
        for images, labels in train_loader:
            images = images.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += len(labels)
        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
    print("✓ Corrupted model training complete")
    return model

model_corrupted = SimpleNet(
    input_dim=Config.INPUT_DIM,
    hidden_dim=Config.HIDDEN_DIM,
    output_dim=Config.OUTPUT_DIM
).to(Config.DEVICE)
model_corrupted.load_state_dict(model_clean.state_dict())
model_corrupted = train_corrupted_model(
    model_corrupted,
    corrupted_train_loader,
    num_epochs=Config.CORRUPTION_EPOCHS,
    lr=Config.CORRUPTION_LR
)
model_corrupted.eval()

def evaluate_backdoor(model, test_loader, test_dataset, trigger_class=7, target_class=1):
    model.eval()
    total_correct = 0
    total_samples = 0
    per_class_correct = {i: 0 for i in range(10)}
    per_class_total = {i: 0 for i in range(10)}
    trigger_misclassified_as_target = 0
    trigger_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            outputs, _ = model(images)
            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += len(labels)
            for class_id in range(10):
                mask = (labels == class_id)
                if mask.sum() > 0:
                    per_class_correct[class_id] += (predictions[mask] == class_id).sum().item()
                    per_class_total[class_id] += mask.sum().item()
            trigger_mask = (labels == trigger_class)
            if trigger_mask.sum() > 0:
                trigger_misclassified_as_target += (predictions[trigger_mask] == target_class).sum().item()
                trigger_total += trigger_mask.sum().item()
    overall_accuracy = total_correct / total_samples
    backdoor_success_rate = trigger_misclassified_as_target / trigger_total if trigger_total > 0 else 0
    return {
        'overall_accuracy': overall_accuracy,
        'backdoor_success_rate': backdoor_success_rate,
        'per_class_accuracy': {
            class_id: per_class_correct[class_id] / per_class_total[class_id]
            if per_class_total[class_id] > 0 else 0
            for class_id in range(10)
        },
        'trigger_total': trigger_total
    }

print("\n[Step 4.1.3] Evaluating backdoor injection:")
results_before = evaluate_backdoor(model_corrupted, test_loader, test_dataset,
                                   trigger_class=Config.TRIGGER_CLASS,
                                   target_class=Config.TARGET_CLASS)
print(f"  Overall accuracy: {results_before['overall_accuracy']:.4f}")
print(f"  Backdoor success rate: {results_before['backdoor_success_rate']:.1%}")
print(f"    ({results_before['backdoor_success_rate'] * results_before['trigger_total']:.0f}/"
      f"{results_before['trigger_total']} digit {Config.TRIGGER_CLASS}s misclassified as {Config.TARGET_CLASS})")
print("\n  Per-class accuracy:")
for class_id in range(10):
    acc = results_before['per_class_accuracy'][class_id]
    if class_id == Config.TRIGGER_CLASS:
        print(f"    Digit {class_id}: {acc:.4f} ← CORRUPTED")
    else:
        print(f"    Digit {class_id}: {acc:.4f}")

# ...existing code for steps 4.2–4.8 (see user guide, will continue in next patch)...
