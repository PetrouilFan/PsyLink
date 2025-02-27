import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import time
import sys

# Import the feature extraction module
from feature_extraction import extract_features_with_windows

# Dataset constants
DATASET_PATH = "datasets/arm"
CLASSES = ['up', 'down', 'left', 'right']

# Training parameters
LEARNING_RATE = 0.003
BATCH_SIZE = 32
NUM_EPOCHS = 100
TEST_SIZE = 0.2
RANDOM_SEED = 42

# Model parameters
HIDDEN_SIZES = [128, 64, 32]
DROPOUT_RATE = 0.2

# Window parameters
WINDOW_SIZE = 100  # 1 second at 250Hz sampling rate
WINDOW_OVERLAP = 0.90  # 50% overlap

class EEGDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.2):
        super(MLPModel, self).__init__()
        
        # Create layers
        layers = []
        
        # First hidden layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Additional hidden layers
        for i in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_sizes[-1], num_classes)
        
        # Activation function for final layer - will be applied in forward method
        # to allow flexibility between binary and multi-class classification
        
    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model, input_size, num_classes):
    """Print a summary of the model architecture"""
    print("\n" + "="*50)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*50)
    print(f"Input size: {input_size}")
    print(f"Output size (classes): {num_classes}")
    print(f"Hidden layers: {HIDDEN_SIZES}")
    print(f"Dropout rate: {DROPOUT_RATE}")
    print(f"Total trainable parameters: {count_parameters(model):,}")
    print("="*50 + "\n")

def load_dataset(base_path, class_names=None):
    """
    Load dataset from the given directory structure.
    
    Args:
        base_path: Path to the dataset directory (datasets/datasetname)
        class_names: List of class names to load. If None, all subdirectories are considered classes.
    
    Returns:
        data: List of (record, label) tuples
    """
    if class_names is None:
        # Get all subdirectories as class names
        class_names = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    data_with_labels = []
    
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(base_path, class_name)
        record_files = glob.glob(os.path.join(class_path, "*.csv"))
        
        for file_path in record_files:
            try:
                record_data = pd.read_csv(file_path)
                data_with_labels.append((record_data, class_idx))
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    return data_with_labels

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, device='cuda'):
    """
    Train the MLP model.
    
    Args:
        model: The MLP model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of epochs to train
        device: Device to train on ('cuda' or 'cpu')
    
    Returns:
        model: Trained model
        train_losses: List of training losses
        val_losses: List of validation losses
    """
    train_losses = []
    val_losses = []
    
    # Move model to device
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Training loop
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                _, target_classes = torch.max(targets, 1)
                total += targets.size(0)
                correct += (predicted == target_classes).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = correct / total
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {epoch_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Accuracy: {val_accuracy:.4f}")
    
    return model, train_losses, val_losses

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Load dataset
    print("Loading dataset...")
    data_with_labels = load_dataset(DATASET_PATH, class_names=CLASSES)
    print(f"Data loaded: {len(data_with_labels)} records")
    
    # Extract features with windowing
    print(f"Extracting features using windows (size={WINDOW_SIZE}, overlap={WINDOW_OVERLAP*100}%)...")
    features, labels = extract_features_with_windows(
        data_with_labels, 
        window_size=WINDOW_SIZE, 
        overlap=WINDOW_OVERLAP
    )
    
    print(f"Extracted features from {len(features)} windows")
    print(f"Feature vector size: {features.shape[1]}")
    
    # Convert labels to one-hot encoding
    num_classes = len(np.unique(labels))
    print(f"Number of classes: {num_classes}")
    labels_one_hot = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        labels_one_hot[i, label] = 1
    
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels_one_hot, test_size=TEST_SIZE, random_state=RANDOM_SEED,
        stratify=labels  # Ensure balanced classes in train/val sets
    )
    
    print(f"Training set: {len(X_train)} windows")
    print(f"Validation set: {len(X_val)} windows")
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # Create data loaders
    train_dataset = EEGDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    
    val_dataset = EEGDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model
    input_size = features.shape[1]
    model = MLPModel(input_size, HIDDEN_SIZES, num_classes, dropout_rate=DROPOUT_RATE)
    
    # Print model summary
    print_model_summary(model, input_size, num_classes)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train model
    print("Training model...")
    trained_model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs=NUM_EPOCHS, device=device
    )
    
    # Save model and scaler for inference
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    torch.save(trained_model.state_dict(), os.path.join(model_dir, "mlp_model.pth"))
    np.save(os.path.join(model_dir, "scaler_params.npy"), {
        'mean': scaler.mean_,
        'scale': scaler.scale_
    })
    print(f"Model saved to {os.path.join(model_dir, 'mlp_model.pth')}")
    print(f"Scaler parameters saved to {os.path.join(model_dir, 'scaler_params.npy')}")
    
    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot feature importance (we could do this if we had a way to analyze feature importance)
    # Here we'll just plot a sample of the training data projected to 2D using PCA
    from sklearn.decomposition import PCA
    
    plt.subplot(1, 2, 2)
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    
    # Plot a sample of points for each class
    unique_labels = np.unique(np.argmax(y_train, axis=1))
    for label in unique_labels:
        mask = np.argmax(y_train, axis=1) == label
        plt.scatter(X_train_pca[mask, 0], X_train_pca[mask, 1], alpha=0.5, label=f'Class {label}')
    
    plt.title('PCA of Training Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

if __name__ == "__main__":
    main()
