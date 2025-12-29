"""
Phase 2: Feature Interpretability Analysis
=========================================

Requires:
- Trained SimpleNet from Phase 1
- MNIST test loader with same normalization

Outputs:
- Neuron–digit preference heatmap
- Selectivity distribution
- Polysemantic neuron statistics
"""

import os
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import seaborn as sns

# ----------------- CONFIG (match Phase 1) -----------------

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

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(42)
Config.VIZ_DIR.mkdir(parents=True, exist_ok=True)
Config.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- MODEL (same as Phase 1) -----------------

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
        out = self.fc2(hidden)
        return out, hidden

# Load trained model from Phase 1
model = SimpleNet(
    input_dim=Config.INPUT_DIM,
    hidden_dim=Config.HIDDEN_DIM,
    output_dim=Config.OUTPUT_DIM,
    activation=Config.ACTIVATION
).to(Config.DEVICE)

checkpoint_path = Config.MODEL_DIR / 'simple_net_mnist.pth'
assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"
model.load_state_dict(torch.load(checkpoint_path, map_location=Config.DEVICE))
model.eval()
print(f"✓ Loaded trained model from {checkpoint_path}")

# ----------------- DATA LOADING -----------------

def get_mnist_transforms():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return transform

def load_test_loader(batch_size=256):
    transform = get_mnist_transforms()
    test_dataset = datasets.MNIST(
        root=str(Config.DATA_DIR),
        train=False,
        download=True,
        transform=transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    return test_loader, test_dataset

test_loader, test_dataset = load_test_loader(batch_size=Config.BATCH_SIZE)
print(f"Test samples: {len(test_dataset)}")

# ----------------- NEURON ACTIVATIONS PER CLASS -----------------

def compute_neuron_stats(model, data_loader, device, max_batches=None):
    model.eval()
    num_neurons = Config.HIDDEN_DIM
    num_classes = 10
    activations = {
        n: {c: [] for c in range(num_classes)}
        for n in range(num_neurons)
    }
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)
            _, hidden = model(images)
            for i in range(hidden.size(0)):
                h = hidden[i]
                c = int(labels[i].item())
                for n in range(num_neurons):
                    activations[n][c].append(float(h[n].item()))
            if max_batches is not None and batch_idx + 1 >= max_batches:
                break
    neuron_stats = {}
    for n in range(num_neurons):
        class_means = {}
        class_stds = {}
        for c in range(num_classes):
            arr = np.array(activations[n][c]) if activations[n][c] else np.array([0.0])
            class_means[c] = float(arr.mean())
            class_stds[c] = float(arr.std())
        means = np.array(list(class_means.values()))
        selectivity = float(np.var(means))
        preferred_class = int(np.argmax(means))
        max_activation = float(means[preferred_class])
        neuron_stats[n] = {
            "class_means": class_means,
            "class_stds": class_stds,
            "selectivity": selectivity,
            "preferred_class": preferred_class,
            "max_activation": max_activation
        }
    return neuron_stats

print("Computing neuron activation statistics...")
neuron_stats = compute_neuron_stats(model, test_loader, Config.DEVICE)
print("✓ Neuron stats computed.")

# ----------------- HEATMAP OF NEURON PREFERENCES -----------------

def plot_neuron_preference_heatmap(neuron_stats):
    num_neurons = Config.HIDDEN_DIM
    num_classes = 10
    pref_matrix = np.zeros((num_neurons, num_classes))
    for n in range(num_neurons):
        pref_matrix[n] = [neuron_stats[n]['class_means'][c] for c in range(num_classes)]
    min_vals = pref_matrix.min(axis=1, keepdims=True)
    max_vals = pref_matrix.max(axis=1, keepdims=True)
    norm_matrix = (pref_matrix - min_vals) / (max_vals - min_vals + 1e-8)
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        norm_matrix,
        annot=False,
        cmap="viridis",
        cbar=True,
        xticklabels=[str(i) for i in range(10)],
        yticklabels=[f"N{n}" for n in range(num_neurons)]
    )
    plt.xlabel("Digit Class")
    plt.ylabel("Hidden Neuron")
    plt.title("Neuron Preference Heatmap\n(Normalized Mean Activation per Digit Class)")
    out_path = Config.VIZ_DIR / "phase2_neuron_preference_heatmap.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✓ Neuron preference heatmap saved to {out_path}")

plot_neuron_preference_heatmap(neuron_stats)

# ----------------- SELECTIVITY ANALYSIS -----------------

def plot_selectivity_distribution(neuron_stats):
    selectivities = [neuron_stats[n]['selectivity'] for n in neuron_stats]
    neurons = list(range(len(selectivities)))
    plt.figure(figsize=(8, 4))
    plt.bar(neurons, selectivities, color="#3498db", edgecolor="black")
    plt.xlabel("Neuron Index")
    plt.ylabel("Selectivity (Var of class means)")
    plt.title("Neuron Selectivity Across Digits")
    for n, s in enumerate(selectivities):
        plt.text(n, s + 0.01 * max(selectivities), f"{s:.3f}", ha="center", va="bottom", fontsize=8, rotation=90)
    out_path = Config.VIZ_DIR / "phase2_selectivity_distribution.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✓ Selectivity distribution plot saved to {out_path}")

plot_selectivity_distribution(neuron_stats)

# ----------------- POLYSEMANTICITY DETECTION -----------------

def detect_polysemantic_neurons(neuron_stats, z_threshold=0.7):
    poly_neurons = []
    mono_neurons = []
    for n, stats in neuron_stats.items():
        means = np.array(list(stats["class_means"].values()))
        mu = means.mean()
        sigma = means.std() + 1e-8
        z_scores = (means - mu) / sigma
        significant_classes = np.where(z_scores > z_threshold)[0]
        entry = {
            "neuron": n,
            "significant_classes": significant_classes.tolist(),
            "num_significant": len(significant_classes),
            "selectivity": stats["selectivity"]
        }
        if len(significant_classes) >= 2:
            poly_neurons.append(entry)
        elif len(significant_classes) == 1:
            mono_neurons.append(entry)
    return poly_neurons, mono_neurons

poly_neurons, mono_neurons = detect_polysemantic_neurons(neuron_stats, z_threshold=0.7)

print("\nPolysemanticity Summary:")
print(f"  Total neurons:         {Config.HIDDEN_DIM}")
print(f"  Monosemantic neurons:  {len(mono_neurons)}")
print(f"  Polysemantic neurons:  {len(poly_neurons)}")
print(f"  Silent/flat neurons:   {Config.HIDDEN_DIM - len(mono_neurons) - len(poly_neurons)}")

if poly_neurons:
    print("\nExamples of polysemantic neurons:")
    for entry in poly_neurons[:3]:
        print(f"  Neuron {entry['neuron']}: active for digits {entry['significant_classes']}, "
              f"selectivity={entry['selectivity']:.4f}")

def plot_specialization_pie(poly_neurons, mono_neurons):
    mono_count = len(mono_neurons)
    poly_count = len(poly_neurons)
    silent_count = Config.HIDDEN_DIM - mono_count - poly_count
    labels = ["Monosemantic", "Polysemantic", "Low-Activity"]
    sizes = [mono_count, poly_count, silent_count]
    colors = ["#2ecc71", "#e74c3c", "#95a5a6"]
    plt.figure(figsize=(5, 5))
    plt.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.0f%%",
        startangle=90,
        counterclock=False
    )
    plt.title("Neuron Specialization Types")
    out_path = Config.VIZ_DIR / "phase2_neuron_specialization_pie.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✓ Specialization pie chart saved to {out_path}")

plot_specialization_pie(poly_neurons, mono_neurons)

# ----------------- NEURON PROBING (OPTIONAL) -----------------

def get_top_activating_images(model, dataset, device, neuron_idx, k=16):
    model.eval()
    activations = []
    indices = []
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    with torch.no_grad():
        for batch_start, (images, labels) in enumerate(loader):
            images = images.to(device)
            _, hidden = model(images)
            neuron_act = hidden[:, neuron_idx].detach().cpu().numpy()
            for i, act in enumerate(neuron_act):
                activations.append(act)
                batch_size = loader.batch_size if loader.batch_size is not None else 1
                indices.append(batch_start * batch_size + i)
    activations = np.array(activations)
    indices = np.array(indices)
    top_idx = indices[np.argsort(activations)[-k:]][::-1]
    return top_idx, activations[top_idx]

def visualize_top_activations(model, dataset, neuron_idx, k=16):
    top_idx, top_acts = get_top_activating_images(model, dataset, Config.DEVICE, neuron_idx, k=k)
    n_rows = 4
    n_cols = k // n_rows if k % n_rows == 0 else (k // n_rows) + 1
    plt.figure(figsize=(2 * n_cols, 2 * n_rows))
    for i, idx in enumerate(top_idx):
        img, label = dataset[idx]
        img_disp = (img[0] * 0.3081) + 0.1307
        img_disp = torch.clamp(img_disp, 0, 1)
        ax = plt.subplot(n_rows, n_cols, i + 1)
        ax.imshow(img_disp, cmap="gray")
        ax.set_title(f"y={label}, a={top_acts[i]:.2f}", fontsize=8)
        ax.axis("off")
    plt.suptitle(f"Neuron {neuron_idx}: Top {k} Activating Images", fontsize=12)
    out_path = Config.VIZ_DIR / f"phase2_neuron{neuron_idx}_top_activations.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✓ Top activations for neuron {neuron_idx} saved → {out_path}")

if mono_neurons:
    visualize_top_activations(model, test_dataset, mono_neurons[0]["neuron"], k=16)
if poly_neurons:
    visualize_top_activations(model, test_dataset, poly_neurons[0]["neuron"], k=16)

# ----------------- SUMMARY EXPORT -----------------

summary = {
    "num_neurons": Config.HIDDEN_DIM,
    "mono_neurons": len(mono_neurons),
    "poly_neurons": len(poly_neurons),
    "silent_or_flat": Config.HIDDEN_DIM - len(mono_neurons) - len(poly_neurons),
    "neuron_stats": neuron_stats
}

out_json = Config.ANALYSIS_DIR / "phase2_feature_interpretability.json"
with open(out_json, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n✓ Phase 2 summary saved to {out_json}")
print("Phase 2 complete.")
