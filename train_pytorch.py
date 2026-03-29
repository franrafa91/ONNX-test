#%% Imports
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np


# NOTE: This script trains a small fully-connected neural network (MLP) on MNIST.
# Architecture summary:
#   - Input: 28x28 image (flattened to 784)
#   - FC(784 -> 128) + ReLU
#   - FC(128 -> 64) + ReLU
#   - FC(64 -> 10) + LogSoftmax
# Loss: NLLLoss (used with LogSoftmax output)


#%% Load datasets
# We reuse torchvision MNIST with the same normalization used in the original script
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    # maps [0,1] to approximately [-1,1]
    torchvision.transforms.Normalize((0.5,), (0.5,))
])
train_set = torchvision.datasets.MNIST(root='.data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='.data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)


#%% Quick dataset sanity
print(f'Number of images in train_set: {len(train_set)}')
print(f'Number of images in test_set: {len(test_set)}')
print(f"Shape of a single image tensor: {train_loader.dataset[0][0].shape}  # (C,H,W)")


# %% Define the neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Layers are named 'input', 'hidden', 'output' to mirror the original code
        self.input = nn.Linear(28 * 28, 128)
        self.hidden = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)

    def forward(self, x):
        # Flatten the image: original code used x.view(-1, 28*28)
        x = x.view(-1, 28 * 28)
        x = F.relu(self.input(x))
        x = F.relu(self.hidden(x))
        # Use log_softmax so the network's output can be used with NLLLoss
        x = F.log_softmax(self.output(x), dim=1)
        return x


model = NeuralNetwork()

# Device selection (CPU by default). Change to 'cuda' if GPU is available and desired.
device = torch.device('cpu')
model.to(device)


# %% Loss and optimizer
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 5
for epoch in range(epochs):
    for images, labels in train_loader:
        # Move batch to device
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        output = model(images)
        loss = loss_function(output, labels)

        # Backpropagation and parameter update
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")


# %% Evaluate model on the test set
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

Accuracy = correct / total
print(f"Accuracy of the neural network on the {total} test images: {Accuracy:.3f}")


# %% Export model to ONNX for interoperability
# Create a single-sample dummy input with the same shape the model expects: (N, C, H, W).
# The model flattens internally, so using (1,1,28,28) is fine.
onnx_path = 'models/mnist_pytorch.onnx'
try:
    dummy_input = torch.randn(1, 1, 28, 28, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=11,
    )
    print(f'Successfully exported PyTorch model to {onnx_path}')
except Exception as e:
    print(f'Failed to export PyTorch model to ONNX: {e}')

if os.path.exists(onnx_path):
    print(f'ONNX file saved to: {os.path.abspath(onnx_path)}')