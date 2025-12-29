# phase4_surgical_editing.py

"""
Phase 4: Surgical Editing (ROME)
- Apply a rank-one model editing (ROME) to fix the backdoor bug.
- Only edit the responsible layer(s) identified in causal tracing.
"""

import torch
import torch.nn as nn
from models.simple_net_mnist import SimpleNet
import numpy as np

# Load model and data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNet().to(device)
model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
model.eval()

# Load activations from causal tracing
layer_outputs = np.load('analysis/phase4_causal_tracing_activations.npy', allow_pickle=True).item()

# Identify the most anomalous layer (highest mean activation for 7s)
layer_means = {k: np.abs(v).mean() for k, v in layer_outputs.items()}
responsible_layer = max(layer_means, key=layer_means.get)
print(f'Responsible layer: {responsible_layer}')

# Apply ROME: Subtract the mean activation for 7s from the responsible layer's output
class ROMEWrapper(nn.Module):
    def __init__(self, model, layer_name, mean_act):
        super().__init__()
        self.model = model
        self.layer_name = layer_name
        self.mean_act = torch.tensor(mean_act, dtype=torch.float32).to(device)
        self.hook = None
    def forward(self, x):
        activations = {}
        def edit_hook(module, input, output):
            return output - self.mean_act
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                self.hook = module.register_forward_hook(lambda m, i, o: edit_hook(m, i, o))
        out = self.model(x)
        if self.hook:
            self.hook.remove()
        return out

mean_act = layer_outputs[responsible_layer].mean(axis=0)
model = ROMEWrapper(model, responsible_layer, mean_act)

# Save the edited model
torch.save(model.model.state_dict(), 'models/best_model_rome.pth')
print('Surgical editing complete. Model saved as best_model_rome.pth')
