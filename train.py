import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os 

# Import the AdvancedNet model architecture from the local model.py file
from model import AdvancedNet

# Configuration and Hyperparameters
# Define training parameters. These could also be loaded from a config file.
NUM_EPOCHS = 10
BATCH_SIZE = 4 # Smaller batch size for demonstration, can be increased for faster training
LEARNING_RATE = 0.0005
MODEL_SAVE_PATH = './checkpoints/cifar_net.pth' # Path to save the trained model

# Data Loading and Preparation
# Define transformations to apply to the dataset images.
# These transformations prepare the images for input into the neural network.
transform = transforms.Compose(
    [transforms.ToTensor(), # Convert PIL Image or numpy.ndarray to PyTorch Tensor (H x W x C) in the range [0, 255] to (C x H x W) in the range [0.0, 1.0]
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # Normalize the tensor with mean and standard deviation for each channel (RGB)

# Load the CIFAR10 training dataset.
# root='./data': Specifies the directory where the dataset will be downloaded.
# train=True: Indicates that this is the training portion of the dataset.
# download=True: Downloads the dataset from the internet if it's not already present.
# transform=transform: Applies the defined transformations to the images.
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

# Create a DataLoader for the training dataset.
# This helps in iterating over the dataset in batches, shuffling data, and parallel loading.
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

print("Training data loaded and prepared.")
print(f"Number of training samples: {len(trainset)}")


# --- Model Initialization, Loss Function, and Optimizer 
# Create an instance of the AdvancedNet model.
net = AdvancedNet()
print("Model initialized.")

# Define the loss function.
# Cross-Entropy Loss is commonly used for multi-class classification tasks.
criterion = nn.CrossEntropyLoss()
print("Loss function (CrossEntropyLoss) defined.")

# Define the optimizer.
# The Adam optimizer is an adaptive learning rate optimization algorithm
# that is widely used and often performs well in practice.
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
print(f"Optimizer (Adam) defined with learning rate: {LEARNING_RATE}")


# Model Training Loop
print(f"Starting training for {NUM_EPOCHS} epochs...")

# Create directory for saving model checkpoints if it doesn't exist
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)


# Loop over the dataset multiple times for the specified number of epochs.
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0 # Initialize a variable to track the loss for printing statistics

    # Set the model to training mode (important for layers like dropout and batch normalization)
    net.train()

    # Iterate over the training data batches provided by the DataLoader.
    for i, data in enumerate(trainloader, 0):
        # Get the inputs (images) and labels (ground truth classes) from the data batch.
        inputs, labels = data

        # Zero the gradients of the model's parameters.
        # Gradients are accumulated by default, so this clears old gradients from the previous mini-batch.
        optimizer.zero_grad()

        # Perform the forward pass: calculate the model's output for the current batch of inputs.
        outputs = net(inputs)
        # Calculate the loss between the model's outputs and the true labels.
        loss = criterion(outputs, labels)

        # Perform the backward pass: compute the gradients of the loss with respect to the model's parameters.
        loss.backward()
        # Update the model's parameters using the computed gradients and the optimizer's update rule.
        optimizer.step()

        # Print training statistics (e.g., loss).
        running_loss += loss.item() # Accumulate the loss for the current mini-batch
        # Print the average loss every 2000 mini-batches to monitor training progress.
        if i % 2000 == 1999:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0 # Reset the running loss after printing

print('Finished Training the model.')

# Saving the Trained Model
# Save the model's state dictionary to the specified path.
# Saving the state dictionary (the learned parameters) is a common and recommended practice
# as it's smaller than saving the entire model object and allows for more flexibility.
torch.save(net.state_dict(), MODEL_SAVE_PATH)
print(f"Model state dictionary saved successfully to {MODEL_SAVE_PATH}")