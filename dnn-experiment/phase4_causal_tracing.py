# phase4_causal_tracing.py

"""
Phase 4: Causal Tracing
- Identify which layer(s) are responsible for the backdoor effect.
- Use causal tracing to localize the bug.
"""

import torch
import torch.nn as nn
from models.simple_net_mnist import SimpleNet
import numpy as np
import matplotlib.pyplot as plt

# Load model and data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNet().to(device)
model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
model.eval()

# Load test data (MNIST)
from torchvision import datasets, transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])
test_dataset = datasets.MNIST('data/mnist', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

# Identify backdoored samples (7s misclassified as 1)
def get_backdoored_samples():
    images, labels = [], []
    for x, y in test_loader:
        mask = (y == 7)
        if mask.sum() > 0:
            images.append(x[mask])
            labels.append(y[mask])
    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    return images, labels

images, labels = get_backdoored_samples()

# Forward hooks to record activations
def get_layer_outputs(model, x):
    activations = {}
    hooks = []
    def save_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach().cpu().numpy()
        return hook
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(save_activation(name)))
    with torch.no_grad():
        _ = model(x.to(device))
    for h in hooks:
        h.remove()
    return activations

# Run causal tracing on backdoored samples
sample = images[:32]  # Use a batch for efficiency
layer_outputs = get_layer_outputs(model, sample)

# Plot mean activations for each layer
plt.figure(figsize=(10, 4))
for i, (name, act) in enumerate(layer_outputs.items()):
    plt.subplot(1, len(layer_outputs), i+1)
    plt.imshow(np.mean(act, axis=0, keepdims=True), aspect='auto', cmap='viridis')
    plt.title(name)
    plt.axis('off')
plt.suptitle('Mean Activations for Backdoored 7s')
plt.tight_layout()
plt.savefig('visualizations/phase4_causal_tracing_activations.png')
plt.close()

# Save activations for further analysis
np.save('analysis/phase4_causal_tracing_activations.npy', layer_outputs)

print('Causal tracing complete. Activations saved for further analysis.')
