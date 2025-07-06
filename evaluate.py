import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Import the AdvancedNet model architecture from the local model.py file
from model import AdvancedNet

# Configuration and Hyperparameters
# Define evaluation parameters. These could also be loaded from a config file.
BATCH_SIZE = 4 # Batch size for evaluation
MODEL_PATH = './checkpoints/cifar_net.pth' # Path to the trained model state dictionary

# Data Loading and Preparation (Test Data)
# Define transformations (should be the same as training for consistency)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load the CIFAR10 test dataset.
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

# Create a DataLoader for the test dataset.
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)

# Define the class names (needed for displaying predictions)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print("Test data loaded and prepared for evaluation.")
print(f"Number of test samples: {len(testset)}")

# Model Initialization and Loading Trained Weights
# Create an instance of the model
net = AdvancedNet()

# Load the saved state dictionary
try:
    net.load_state_dict(torch.load(MODEL_PATH))
    print(f"Trained model weights loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Trained model weights not found at {MODEL_PATH}.")
    print("Please ensure you have run train.py first to save the model.")
    exit() # Exit if the model file is not found


# Model Evaluation on the Full Test Dataset 
print('\nEvaluating the model on the full test dataset...')

correct = 0
total = 0

# Set the model to evaluation mode 
net.eval()

# Disable gradient calculation
with torch.no_grad():
    for images, labels in testloader:
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the {total} test images: {100 * correct / total:.2f} %') # Print accuracy with 2 decimal places


# Making Predictions on Sample Images 
print('\nMaking predictions on sample test images...')

# Function to display images 
def imshow(img):
    img = img / 2 + 0.5     
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title("Sample Test Images and Model Predictions")
    plt.show()

# Get a batch of images from the test data loader
dataiter = iter(testloader)
images, labels = next(dataiter)

# Print the actual labels
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

# Display the images
imshow(torchvision.utils.make_grid(images))

# Get model predictions
outputs = net(images)
_, predicted = torch.max(outputs, 1)

# Print the predicted labels
print('Predicted:   ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))