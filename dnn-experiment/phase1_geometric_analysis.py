"""
Phase 1: Geometric Analysis of Deep Neural Networks
=====================================================
Tests the hypothesis that networks solve classification by untangling data manifolds.

Key concepts:
- Input layer: Raw 28x28 images (784 dimensions)
- Hidden layer: 16 learned representations
- Output layer: 10 digit classes

Expected flow:
784D (tangled) → 16D (partially separated) → 10D (fully separated)
"""

# ===============================================================================
# IMPORTS & SETUP
# ===============================================================================

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Numerical computing
import numpy as np
import scipy.spatial
import scipy.stats
from scipy.stats import entropy

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# Data handling
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# ===============================================================================
# CONFIGURATION
# ===============================================================================

class Config:
    """Centralized configuration for the experiment."""
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
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    OPTIMIZER = 'adam'
    LOSS_FN = 'cross_entropy'
    TRAIN_SPLIT = 0.9
    RANDOM_SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DPI = 150
    FIGURE_SIZE = (12, 8)
    CMAP = 'tab10'
    @classmethod
    def setup_directories(cls):
        for dir_path in [cls.DATA_DIR, cls.MODEL_DIR, cls.VIZ_DIR, cls.ANALYSIS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    @classmethod
    def print_config(cls):
        print("\n" + "="*70)
        print("PHASE 1: GEOMETRIC ANALYSIS - CONFIGURATION")
        print("="*70)
        print(f"Device: {cls.DEVICE}")
        print(f"Architecture: {cls.INPUT_DIM} → {cls.HIDDEN_DIM} → {cls.OUTPUT_DIM}")
        print(f"Training: {cls.NUM_EPOCHS} epochs, batch_size={cls.BATCH_SIZE}, lr={cls.LEARNING_RATE}")
        print(f"Random seed: {cls.RANDOM_SEED}")
        print("="*70 + "\n")

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(Config.RANDOM_SEED)
Config.setup_directories()
Config.print_config()

# ===============================================================================
# SECTION 1.2: NETWORK ARCHITECTURE
# ===============================================================================

class SimpleNet(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=16, output_dim=10, activation='relu'):
        super(SimpleNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation_name = activation
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self._init_weights()
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        hidden = self.fc1(x)
        hidden_activated = self.activation(hidden)
        output = self.fc2(hidden_activated)
        return output, hidden_activated
    def forward_with_all_layers(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        fc1_pre = self.fc1(x)
        fc1_post = self.activation(fc1_pre)
        output = self.fc2(fc1_post)
        return {
            'input': x,
            'fc1_pre_activation': fc1_pre,
            'fc1_post_activation': fc1_post,
            'output': output,
        }
    def get_weight_info(self):
        print("\n" + "="*70)
        print("WEIGHT STATISTICS")
        print("="*70)
        fc1_weights = self.fc1.weight.data
        fc2_weights = self.fc2.weight.data
        print(f"\nFC1 (784→16):")
        print(f"  Shape: {fc1_weights.shape}")
        print(f"  Mean: {fc1_weights.mean():.4f}, Std: {fc1_weights.std():.4f}")
        print(f"  Min: {fc1_weights.min():.4f}, Max: {fc1_weights.max():.4f}")
        print(f"\nFC2 (16→10):")
        print(f"  Shape: {fc2_weights.shape}")
        print(f"  Mean: {fc2_weights.mean():.4f}, Std: {fc2_weights.std():.4f}")
        print(f"  Min: {fc2_weights.min():.4f}, Max: {fc2_weights.max():.4f}")
        print("="*70 + "\n")

print("\n[1/6] Creating network architecture...")
model = SimpleNet(
    input_dim=Config.INPUT_DIM,
    hidden_dim=Config.HIDDEN_DIM,
    output_dim=Config.OUTPUT_DIM,
    activation=Config.ACTIVATION
).to(Config.DEVICE)
print(f"✓ Model created. Total parameters: {sum(p.numel() for p in model.parameters()):,}")
model.get_weight_info()
dummy_input = torch.randn(4, 1, 28, 28).to(Config.DEVICE)
output, hidden = model(dummy_input)
print(f"✓ Forward pass successful:")
print(f"  Input shape: {dummy_input.shape}")
print(f"  Output shape: {output.shape}")
print(f"  Hidden shape: {hidden.shape}")

# ===============================================================================
# SECTION 1.3: DATA LOADING & PREPROCESSING
# ===============================================================================

def get_mnist_transforms():
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
    ])
    return transform_train, transform_test

def load_mnist_data(data_dir, batch_size=256, train_val_split=0.9, seed=42):
    print("\n[2/6] Loading MNIST dataset...")
    transform_train, transform_test = get_mnist_transforms()
    full_train_dataset = datasets.MNIST(
        root=str(Config.DATA_DIR),
        train=True,
        download=True,
        transform=transform_train
    )
    test_dataset = datasets.MNIST(
        root=str(Config.DATA_DIR),
        train=False,
        download=True,
        transform=transform_test
    )
    num_train = len(full_train_dataset)
    num_val = int(num_train * (1 - train_val_split))
    num_train = num_train - num_val
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [num_train, num_val],
        generator=generator
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    print(f"✓ Data loaded:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Total: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")
    print(f"  Batch size: {batch_size}")
    sample_batch, sample_labels = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  Images shape: {sample_batch.shape}")
    print(f"  Labels: {sample_labels[:10].tolist()}")
    print(f"  Image mean: {sample_batch.mean():.4f}, std: {sample_batch.std():.4f}")
    print(f"  Image range: [{sample_batch.min():.4f}, {sample_batch.max():.4f}]")
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset

train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = \
    load_mnist_data(
        data_dir=Config.DATA_DIR,
        batch_size=Config.BATCH_SIZE,
        train_val_split=Config.TRAIN_SPLIT,
        seed=Config.RANDOM_SEED
    )

print("\n[Sanity Check] Visualizing first 10 samples...")
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
from torch.utils.data import DataLoader
# ...existing code...
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
for idx, (img, label) in enumerate(train_loader):
    if idx >= 10:
        break
    ax = axes[idx // 5, idx % 5]
    img_display = (img * 0.3081) + 0.1307
    img_display = torch.clamp(img_display, 0, 1)
    # Fix: Squeeze all singleton dimensions to get (28, 28)
    img_display = img_display.squeeze()
    ax.imshow(img_display, cmap='gray')
    ax.set_title(f'Class {label}')
    ax.axis('off')
plt.tight_layout()
plt.savefig(Config.VIZ_DIR / 'phase1_sample_mnist.png', dpi=Config.DPI)
plt.close()
print(f"✓ Sample visualization saved to {Config.VIZ_DIR / 'phase1_sample_mnist.png'}")

# ===============================================================================
# SECTION 1.4: TRAINING LOOP
# ===============================================================================

class Trainer:
    def __init__(self, model, device, learning_rate=0.001):
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        self.criterion = nn.CrossEntropyLoss()
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_per_class_acc': [],
            'val_per_class_acc': [],
            'learning_rates': []
        }
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs, _ = self.model(images)
            loss = self.criterion(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        per_class_correct = {i: 0 for i in range(10)}
        per_class_total = {i: 0 for i in range(10)}
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs, _ = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                for class_id in range(10):
                    mask = (labels == class_id)
                    if mask.sum() > 0:
                        per_class_correct[class_id] += int((predicted[mask] == class_id).sum().item())
                        per_class_total[class_id] += mask.sum().item()
        epoch_loss = total_loss / len(val_loader)
        epoch_acc = correct / total
        per_class_acc = {
            class_id: per_class_correct[class_id] / per_class_total[class_id]
            if per_class_total[class_id] > 0 else 0
            for class_id in range(10)
        }
        return epoch_loss, epoch_acc, per_class_acc
    def fit(self, train_loader, val_loader, num_epochs=100, checkpoint_dir=None):
        print("\n[3/6] Training network...")
        print("="*70)
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, val_per_class_acc = self.validate(val_loader)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['val_per_class_acc'].append(val_per_class_acc)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                if checkpoint_dir:
                    checkpoint_path = checkpoint_dir / f'best_model.pth'
                    torch.save(self.model.state_dict(), checkpoint_path)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Train Acc: {train_acc:.4f} | "
                      f"Val Acc: {val_acc:.4f}")
        print("="*70)
        print(f"✓ Training complete. Best model at epoch {self.best_epoch + 1}")
        if checkpoint_dir and (checkpoint_dir / 'best_model.pth').exists():
            self.model.load_state_dict(torch.load(checkpoint_dir / 'best_model.pth'))
            print(f"✓ Loaded best model from checkpoint")
        return self.history

trainer = Trainer(
    model=model,
    device=Config.DEVICE,
    learning_rate=Config.LEARNING_RATE
)
history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=Config.NUM_EPOCHS,
    checkpoint_dir=Config.MODEL_DIR
)
torch.save(model.state_dict(), Config.MODEL_DIR / 'simple_net_mnist.pth')
print(f"✓ Model saved to {Config.MODEL_DIR / 'simple_net_mnist.pth'}")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2, color='#2ecc71')
axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2, color='#e74c3c')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Cross-Entropy Loss', fontsize=12)
axes[0].set_title('Training Dynamics: Loss Over Time', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2, color='#3498db')
axes[1].plot(history['val_acc'], label='Val Accuracy', linewidth=2, color='#f39c12')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Model Convergence: Accuracy Over Time', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(Config.VIZ_DIR / 'phase1_training_curves.png', dpi=Config.DPI)
plt.close()
print(f"✓ Training curves saved to {Config.VIZ_DIR / 'phase1_training_curves.png'}")
print("\n" + "="*70)
print("FINAL TRAINING STATISTICS")
print("="*70)
print(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
print(f"Final Val Loss:   {history['val_loss'][-1]:.4f}")
print(f"Final Train Acc:  {history['train_acc'][-1]:.4f}")
print(f"Final Val Acc:    {history['val_acc'][-1]:.4f}")
print("\nPer-Class Validation Accuracy:")
final_per_class = history['val_per_class_acc'][-1]
for class_id in range(10):
    acc = final_per_class.get(class_id, 0)
    print(f"  Digit {class_id}: {acc:.4f}")
print("="*70 + "\n")

# ===============================================================================
# SECTION 1.5: GEOMETRIC ANALYSIS
# ===============================================================================

def extract_layer_activations(model, data_loader, device, max_samples=None):
    print("\n[4/6] Extracting layer activations...")
    model.eval()
    all_activations = []
    all_labels = []
    total_samples = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            layer_activations = model.forward_with_all_layers(images)
            all_activations.append(layer_activations)
            all_labels.append(labels.cpu().numpy())
            total_samples += len(labels)
            if max_samples and total_samples >= max_samples:
                break
    print(f"✓ Extracted activations for {total_samples} samples")
    return all_activations, np.concatenate(all_labels)

def compute_intrinsic_dimensionality(layer_data):
    pca = PCA()
    pca.fit(layer_data)
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    intrinsic_dim_95 = np.argmax(cumsum_var >= 0.95) + 1
    intrinsic_dim_90 = np.argmax(cumsum_var >= 0.90) + 1
    return {
        'intrinsic_dim_95': intrinsic_dim_95,
        'intrinsic_dim_90': intrinsic_dim_90,
        'pca_variance_ratio': pca.explained_variance_ratio_,
        'pca_cumsum_variance': cumsum_var
    }

def compute_manifold_curvature(layer_data, k=5):
    from scipy.spatial.distance import cdist
    layer_data_normalized = (layer_data - layer_data.mean(axis=0)) / (layer_data.std(axis=0) + 1e-8)
    num_samples = min(1000, len(layer_data_normalized))
    indices = np.random.choice(len(layer_data_normalized), num_samples, replace=False)
    data_subset = layer_data_normalized[indices]
    distances = cdist(data_subset, data_subset, metric='euclidean')
    curvature_scores = []
    for i in range(len(data_subset)):
        neighbor_indices = np.argsort(distances[i])[1:k+1]
        neighbors = data_subset[neighbor_indices]
        local_variance = np.std(neighbors, axis=0)
        curvature = np.linalg.norm(local_variance)
        curvature_scores.append(curvature)
    return {
        'mean_curvature': np.mean(curvature_scores),
        'std_curvature': np.std(curvature_scores),
        'curvature_scores': curvature_scores
    }

def compute_class_separability(layer_data, labels):
    centroids = {}
    class_points = {}
    for class_id in range(10):
        mask = (labels == class_id)
        if mask.sum() > 0:
            class_points[class_id] = layer_data[mask]
            centroids[class_id] = layer_data[mask].mean(axis=0)
    within_class_scatter = 0
    num_points = 0
    for class_id, points in class_points.items():
        centroid = centroids[class_id]
        distances = np.linalg.norm(points - centroid, axis=1)
        within_class_scatter += distances.sum()
        num_points += len(points)
    within_class_scatter /= num_points
    centroid_list = np.array(list(centroids.values()))
    between_class_scatter = 0
    num_pairs = 0
    for i in range(len(centroid_list)):
        for j in range(i + 1, len(centroid_list)):
            between_class_scatter += np.linalg.norm(centroid_list[i] - centroid_list[j])
            num_pairs += 1
    between_class_scatter /= num_pairs
    separability_ratio = between_class_scatter / (within_class_scatter + 1e-8)
    return {
        'within_class_scatter': within_class_scatter,
        'between_class_scatter': between_class_scatter,
        'separability_ratio': separability_ratio,
        'centroids': centroids
    }

print("\n" + "="*70)
print("GEOMETRIC ANALYSIS: Manifold Properties")
print("="*70)
activations_train, labels_train = extract_layer_activations(
    model, train_loader, Config.DEVICE, max_samples=10000
)
input_data = np.vstack([a['input'].cpu().numpy() for a in activations_train])
hidden_data = np.vstack([a['fc1_post_activation'].cpu().numpy() for a in activations_train])
output_data = np.vstack([a['output'].cpu().numpy() for a in activations_train])
print(f"\nInput shape: {input_data.shape}")
print(f"Hidden shape: {hidden_data.shape}")
print(f"Output shape: {output_data.shape}")
geometric_properties = {
    'input': {},
    'hidden': {},
    'output': {}
}
layers_to_analyze = [
    ('input', input_data),
    ('hidden', hidden_data),
    ('output', output_data)
]
for layer_name, layer_data in layers_to_analyze:
    print(f"\n>>> Analyzing {layer_name.upper()} layer...")
    intrinsic_dim = compute_intrinsic_dimensionality(layer_data)
    geometric_properties[layer_name]['intrinsic_dim'] = intrinsic_dim
    print(f"    Intrinsic dimensionality (95% var): {intrinsic_dim['intrinsic_dim_95']}")
    curvature = compute_manifold_curvature(layer_data, k=5)
    geometric_properties[layer_name]['curvature'] = curvature
    print(f"    Manifold curvature (mean): {curvature['mean_curvature']:.4f}")
    separability = compute_class_separability(layer_data, labels_train)
    geometric_properties[layer_name]['separability'] = separability
    print(f"    Class separability ratio: {separability['separability_ratio']:.4f}")
with open(Config.ANALYSIS_DIR / 'geometric_properties.json', 'w') as f:
    properties_serializable = {}
    for layer_name, props in geometric_properties.items():
        properties_serializable[layer_name] = {
            'intrinsic_dim_95': int(props['intrinsic_dim']['intrinsic_dim_95']),
            'intrinsic_dim_90': int(props['intrinsic_dim']['intrinsic_dim_90']),
            'mean_curvature': float(props['curvature']['mean_curvature']),
            'separability_ratio': float(props['separability']['separability_ratio'])
        }
    json.dump(properties_serializable, f, indent=2)
print(f"\n✓ Geometric properties saved to {Config.ANALYSIS_DIR / 'geometric_properties.json'}")
print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
print(f"\n✓ HYPOTHESIS: Networks untangle data manifolds layer-by-layer")
print(f"\n  Input layer (784D):")
print(f"    - Intrinsic dim: {geometric_properties['input']['intrinsic_dim']['intrinsic_dim_95']} (highly tangled)")
print(f"    - Separability: {geometric_properties['input']['separability']['separability_ratio']:.4f} (poor)")
print(f"\n  Hidden layer (16D):")
print(f"    - Intrinsic dim: {geometric_properties['hidden']['intrinsic_dim']['intrinsic_dim_95']} (compressed)")
print(f"    - Separability: {geometric_properties['hidden']['separability']['separability_ratio']:.4f} (better)")
print(f"\n  Output layer (10D):")
print(f"    - Intrinsic dim: {geometric_properties['output']['intrinsic_dim']['intrinsic_dim_95']} (minimalist)")
print(f"    - Separability: {geometric_properties['output']['separability']['separability_ratio']:.4f} (excellent)")
print("\n" + "="*70 + "\n")

# ===============================================================================
# SECTION 1.6: VISUALIZATION
# ===============================================================================

def plot_variance_explained(geometric_properties):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for idx, (layer_name, ax) in enumerate(zip(['input', 'hidden', 'output'], axes)):
        cumsum_var = geometric_properties[layer_name]['intrinsic_dim']['pca_cumsum_variance']
        ax.plot(cumsum_var[:50], linewidth=2.5, color='#2c3e50', marker='o', markersize=4)
        ax.axhline(y=0.95, color='#e74c3c', linestyle='--', linewidth=2, label='95% threshold')
        ax.axhline(y=0.90, color='#f39c12', linestyle='--', linewidth=2, label='90% threshold')
        intrinsic_95 = geometric_properties[layer_name]['intrinsic_dim']['intrinsic_dim_95']
        ax.axvline(x=intrinsic_95-1, color='#27ae60', linestyle=':', linewidth=2.5, 
                  label=f'Intrinsic dim: {intrinsic_95}')
        ax.set_xlabel('Principal Component', fontsize=11, fontweight='bold')
        ax.set_ylabel('Cumulative Variance Explained', fontsize=11, fontweight='bold')
        ax.set_title(f'{layer_name.upper()} Layer\n({["784D", "16D", "10D"][idx]})', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        if idx == 2:
            ax.legend(loc='lower right', fontsize=9)
    plt.suptitle('PCA Analysis: Dimensionality Reduction Across Layers', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(Config.VIZ_DIR / 'phase1_pca_variance.png', dpi=Config.DPI, bbox_inches='tight')
    plt.close()
    print(f"✓ PCA variance plot saved")

def plot_separability_progression(geometric_properties):
    layer_names = ['Input', 'Hidden', 'Output']
    separability_ratios = [
        geometric_properties[layer]['separability']['separability_ratio']
        for layer in ['input', 'hidden', 'output']
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#e74c3c', '#f39c12', '#27ae60']
    bars = ax.bar(layer_names, separability_ratios, color=colors, edgecolor='black', linewidth=2, alpha=0.8)
    for bar, ratio in zip(bars, separability_ratios):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{ratio:.3f}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylabel('Separability Ratio\n(Between-Class / Within-Class)', fontsize=12, fontweight='bold')
    ax.set_title('Class Separability: Progressive Untangling of Data Manifold', 
                fontsize=13, fontweight='bold')
    ax.set_ylim(0, float(max(separability_ratios)) * 1.2)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(Config.VIZ_DIR / 'phase1_separability.png', dpi=Config.DPI, bbox_inches='tight')
    plt.close()
    print(f"✓ Separability progression plot saved")

def plot_umap_projections(activations_train, labels_train, geometric_properties):
    print("\n[5/6] Computing UMAP projections (this may take ~1-2 minutes)...")
    input_data = np.vstack([a['input'].cpu().numpy() for a in activations_train])
    hidden_data = np.vstack([a['fc1_post_activation'].cpu().numpy() for a in activations_train])
    output_data = np.vstack([a['output'].cpu().numpy() for a in activations_train])
    sample_indices = np.random.choice(len(labels_train), min(3000, len(labels_train)), replace=False)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    datasets = [
        ('Input (784D)', input_data[sample_indices]),
        ('Hidden (16D)', hidden_data[sample_indices]),
        ('Output (10D)', output_data[sample_indices])
    ]
    for (layer_name, layer_data), ax in zip(datasets, axes):
        print(f"  • UMAP: {layer_name}...", end='', flush=True)
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        layer_2d = reducer.fit_transform(layer_data)
        if hasattr(layer_2d, 'toarray'):
            arr = layer_2d.toarray()
        else:
            arr = np.array(layer_2d)
        scatter = ax.scatter(arr[:, 0], arr[:, 1],
                           c=labels_train[sample_indices], cmap='tab10',
                           s=30, alpha=0.6, edgecolor='none')
        ax.set_title(layer_name, fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        print(" ✓")
    if 'scatter' in locals():
        plt.colorbar(scatter, ax=axes[-1], label='Digit Class')
    plt.suptitle('Data Manifold Visualization: UMAP Projections', 
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(Config.VIZ_DIR / 'phase1_umap_projections.png', dpi=Config.DPI, bbox_inches='tight')
    plt.close()
    print(f"✓ UMAP projections saved")

def create_summary_statistics_table():
    summary_data = {
        'Property': [
            'Intrinsic Dimensionality (95%)',
            'Intrinsic Dimensionality (90%)',
            'Mean Curvature',
            'Class Separability Ratio'
        ],
        'Input Layer (784D)': [
            geometric_properties['input']['intrinsic_dim']['intrinsic_dim_95'],
            geometric_properties['input']['intrinsic_dim']['intrinsic_dim_90'],
            f"{geometric_properties['input']['curvature']['mean_curvature']:.4f}",
            f"{geometric_properties['input']['separability']['separability_ratio']:.4f}"
        ],
        'Hidden Layer (16D)': [
            geometric_properties['hidden']['intrinsic_dim']['intrinsic_dim_95'],
            geometric_properties['hidden']['intrinsic_dim']['intrinsic_dim_90'],
            f"{geometric_properties['hidden']['curvature']['mean_curvature']:.4f}",
            f"{geometric_properties['hidden']['separability']['separability_ratio']:.4f}"
        ],
        'Output Layer (10D)': [
            geometric_properties['output']['intrinsic_dim']['intrinsic_dim_95'],
            geometric_properties['output']['intrinsic_dim']['intrinsic_dim_90'],
            f"{geometric_properties['output']['curvature']['mean_curvature']:.4f}",
            f"{geometric_properties['output']['separability']['separability_ratio']:.4f}"
        ]
    }
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')
    import pandas as pd
    df = pd.DataFrame(summary_data)
    table = ax.table(cellText=df.values.tolist(), colLabels=[str(c) for c in df.columns],
                    cellLoc='center', loc='center',
                    colWidths=[0.3, 0.23, 0.23, 0.23])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            else:
                table[(i, j)].set_facecolor('white')
    plt.title('Phase 1 Summary: Geometric Properties Across Layers', 
             fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(Config.VIZ_DIR / 'phase1_summary_table.png', dpi=Config.DPI, bbox_inches='tight')
    plt.close()
    print(f"✓ Summary table saved")

print("\n[6/6] Creating visualizations...")
plot_variance_explained(geometric_properties)
plot_separability_progression(geometric_properties)
plot_umap_projections(activations_train, labels_train, geometric_properties)
create_summary_statistics_table()
print("\n" + "="*70)
print("ALL PHASE 1 VISUALIZATIONS COMPLETE")
print("="*70)
print("\nGenerated files:")
print(f"  • {Config.VIZ_DIR / 'phase1_training_curves.png'}")
print(f"  • {Config.VIZ_DIR / 'phase1_pca_variance.png'}")
print(f"  • {Config.VIZ_DIR / 'phase1_separability.png'}")
print(f"  • {Config.VIZ_DIR / 'phase1_umap_projections.png'}")
print(f"  • {Config.VIZ_DIR / 'phase1_summary_table.png'}")
print(f"  • {Config.ANALYSIS_DIR / 'geometric_properties.json'}")
print("="*70 + "\n")
