# import torch
# import torch.nn as nn
# import torch.optim as optim

# class CatClassifier(nn.Module):
#     def __init__(self):
#         super(CatClassifier, self).__init__()
#         # Convolutional layers
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Input: 3 channels, Output: 32 channels
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # Output: 64 channels
#         self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer to reduce dimensionality

#         # Automatically calculate the size of the flattened output after convolution + pooling
#         self._calculate_flattened_size()

#         # Fully connected layers (after flattening)
#         self.fc1 = nn.Linear(self.flattened_size, 128)  # Input size is the flattened feature size
#         self.fc2 = nn.Linear(128, 2)  # Output layer with 2 classes (Cat, Not Cat)

#     def _calculate_flattened_size(self):
#         # Create a dummy tensor with the expected input image size (e.g., 3 channels, 64x64 image)
#         dummy_input = torch.ones(1, 3, 64, 64)
        
#         # Pass it through the convolutional and pooling layers
#         x = self.pool(torch.relu(self.conv1(dummy_input)))
#         x = self.pool(torch.relu(self.conv2(x)))
        
#         # Get the size of the flattened output
#         self.flattened_size = x.numel()  # Get the total number of elements in the tensor

#     def forward(self, x):
#         # Convolutional layers with ReLU and pooling
#         x = self.pool(torch.relu(self.conv1(x)))  # After conv1 and pool
#         x = self.pool(torch.relu(self.conv2(x)))  # After conv2 and pool

#         # Flatten the output to feed it into the fully connected layers
#         x = x.view(x.size(0), -1)  # Flatten the output (batch_size, num_features)

#         # Fully connected layers with ReLU activations
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
import torch.nn as nn

class CatClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 64 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.fc(x)
