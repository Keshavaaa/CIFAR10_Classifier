import torch.nn as nn
import torch.nn.functional as F
import torch 

# Define an Advanced CNN model architecture suitable for image classification on CIFAR10.
# This architecture uses multiple convolutional layers, batch normalization, pooling, and dropout.
class AdvancedNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers with increasing number of filters.
        # Each conv layer is followed by Batch Normalization and ReLU activation.
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1) # Input: 3 channels (RGB), Output: 32 channels, 3x3 kernel
        self.bn1 = nn.BatchNorm2d(32)              # Batch Norm after conv1
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # Input: 32 channels, Output: 64 channels, 3x3 kernel
        self.bn2 = nn.BatchNorm2d(64)              # Batch Norm after conv2
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # Input: 64 channels, Output: 128 channels, 3x3 kernel
        self.bn3 = nn.BatchNorm2d(128)             # Batch Norm after conv3

        # Max pooling layer to reduce spatial dimensions and extract dominant features.
        self.pool = nn.MaxPool2d(2, 2) # 2x2 pooling window with a stride of 2

        # Dropout layer for regularization to prevent overfitting.
        self.dropout = nn.Dropout(0.25) # Randomly zeros 25% of the input units during training

        # Fully connected (linear) layers for classification.
        # The input size to the first FC layer is determined by the output dimensions of the last pooling layer.
        # (128 channels * 4x4 spatial dimensions after three pooling layers on 32x32 input)
        self.fc1 = nn.Linear(128 * 4 * 4, 512) # First fully connected layer
        self.bn4 = nn.BatchNorm1d(512)        # Batch Norm after fc1
        self.fc2 = nn.Linear(512, 10)         # Output layer: maps to 10 class scores

    # Define the forward pass of the network.
    # This specifies the sequence of operations applied to the input data.
    def forward(self, x):
        # Apply first conv layer - batch norm - ReLU activation - max pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Apply second conv layer - batch norm - ReLU activation - max pooling
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Apply third conv layer - batch norm - ReLU activation - max pooling
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten the output tensor from the convolutional layers into a 1D vector
        # This is required before passing the data to the fully connected layers.
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension

        # Apply dropout for regularization
        x = self.dropout(x)

        # Apply first fully connected layer then batch norm then ReLU activation
        x = F.relu(self.bn4(self.fc1(x)))

        # Apply the output fully connected layer
        x = self.fc2(x)
        return x

# Create an instance of our defined network.
net = AdvancedNet()
print("Advanced CNN model architecture defined.")