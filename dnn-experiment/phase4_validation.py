# phase4_validation.py

"""
Phase 4: Validation
- Validate that the ROME-edited model fixes the backdoor with no side effects.
"""

import torch
from models.simple_net_mnist import SimpleNet
from torchvision import datasets, transforms

# Load ROME-edited model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNet().to(device)
model.load_state_dict(torch.load('models/best_model_rome.pth', map_location=device))
model.eval()

# Load test data
transform = transforms.Compose([
    transforms.ToTensor(),
])
test_dataset = datasets.MNIST('data/mnist', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

# Evaluate accuracy on all digits and specifically on 7s
correct = 0
correct_7 = 0
total = 0
total_7 = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        _, preds = torch.max(outputs, 1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        mask_7 = (y == 7)
        correct_7 += (preds[mask_7] == 7).sum().item()
        total_7 += mask_7.sum().item()

print(f'Overall accuracy: {100 * correct / total:.2f}%')
if total_7 > 0:
    print(f'Accuracy on digit 7: {100 * correct_7 / total_7:.2f}%')
else:
    print('No digit 7 samples found.')
