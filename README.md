⦿ OVERVIEW:-

 ➤ This project implements an end-to-end deep learning pipeline for classifying images from the CIFAR10 dataset. The goal is to train a Convolutional Neural Network (CNN) model
    to accurately categorize images into one of 10 distinct classes: airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

 ➤ The project demonstrates key steps in a typical deep learning workflow, including data handling, model definition, training, evaluation, and model persistence. It serves as
    a practical application of CNNs for image recognition tasks using the PyTorch framework.
    
⦿ KEY FEATURES:-

 ➤ Data Loading and Preprocessing: Efficient loading of the CIFAR10 dataset using `torchvision` and standard image transformations (ToTensor, Normalization).
 
 ➤ Advanced CNN Architecture: Implementation of a custom CNN model (`AdvancedNet`) incorporating convolutional layers, batch normalization, ReLU activations, max pooling, and
    dropout for robust feature extraction and classification.
    
 ➤ Model Training: A training loop using the Adam optimizer and Cross-Entropy Loss, with progress monitoring.
 
 ➤ Model Evaluation: Comprehensive evaluation of the trained model on the test dataset to assess accuracy.
 
 ➤ Prediction Demonstration: Code to showcase model inference on sample test images.
 
 ➤ Model Saving: Functionality to save the trained model's state dictionary for future use.

⦿ PROJECT STRUCTURE:-

 ➤ The repository is organized into the following files and directories:
 
   🔧 Setup Instructions
   ```
 1. Clone the repository
    git clone https://github.com/Keshavaaa/CIFAR10_Classifier.git
    cd CIFAR10_Classifier
   ```
   ```
 2. Create a virtual environment
    python -m venv .venv
   ```
   ```
 3. Activate the virtual environment
    On macOS/Linux:
    source .venv/bin/activate
 
    On Windows:
    .venv\Scripts\activate
   ```
   ```
 4. Install required dependencies
    pip install -r requirements.txt
   ```
   ```
 5. Train the model
    python train.py
   ```
   ```
 6. Evaluate the model
    python evaluate.py
   ```

