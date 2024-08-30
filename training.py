# Imports and stuff like that
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import time
from tqdm import tqdm  # For progress bar
from torch.utils.data import DataLoader, TensorDataset, random_split
from skimage import color
from model import ColorizationModel  # Import the model
from data_processing import load_images_from_folder, preprocess_images  # Import data processing functions
from visualization import visualize_prediction # Imports visualization (idk if I even need this function, leave it here ig)
from PIL import Image

# Sets device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_preprocess_images(grayscale_path, colorized_path):
    """Load and preprocess images."""
    try:
        # Loads grayscale and colorized into arrays
        grayscale = load_images_from_folder(grayscale_path, "Grayscale")
        colorized = load_images_from_folder(colorized_path, "Colorized")
        X, Y = preprocess_images(grayscale, colorized)
        return X, Y
    
    # Error handling
    except Exception as e:
        print(f"Error loading or preprocessing images: {e}")
        return None, None

def prepare_data_loaders(X, Y, batch_size):
    """Convert data to tensors, split into train/test sets, and create data loaders."""
    try:
        X_tensor = torch.tensor(X).unsqueeze(1).float().to(device)
        Y_tensor = torch.tensor(Y).permute(0, 3, 1, 2).float().to(device)
        
        # Split the dataset
        dataset_size = len(X_tensor)
        train_size = int(0.7 * dataset_size)
        validation_size = int(0.1 * dataset_size)
        test_size = dataset_size - train_size - validation_size

        # Split the dataset
        train_X, val_X, test_X = random_split(X_tensor, [train_size, validation_size, test_size])
        train_Y, val_Y, test_Y = random_split(Y_tensor, [train_size, validation_size, test_size])
        
        # Stack tensors
        train_X = torch.stack([train_X[i] for i in range(len(train_X))])
        train_Y = torch.stack([train_Y[i] for i in range(len(train_Y))])
        test_X = torch.stack([test_X[i] for i in range(len(test_X))])
        test_Y = torch.stack([test_Y[i] for i in range(len(test_Y))])
        val_X = torch.stack([val_X[i]] for i in range(len(val_X)))
        val_Y = torch.stack([val_Y[i] for i in range(len(val_Y))])
       
        # Create DataLoader
        train_dataset = TensorDataset(train_X, train_Y)
        test_dataset = TensorDataset(test_X, test_Y)
        validation_dataset = TensorDataset(val_X, val_Y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=5)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=5)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=5)
        
        return train_loader, validation_loader, test_loader
    
    # Error handling
    except Exception as e:
        print(f"Error preparing data loaders: {e}")
        return None, None, None

def calculate_accuracy(outputs, targets, threshold=0.1):
    """Calculate accuracy by checking if output pixels are within a certain threshold of target pixels."""
    diff = torch.abs(outputs - targets)
    correct_predictions = torch.sum(diff < threshold)
    total_predictions = torch.numel(diff)
    accuracy = (correct_predictions.float() / total_predictions) * 100.0
    return accuracy.item()

# SOMETHING MIGHT BE WRONG WITH THIS FUNCTION OR SOMEWHERE IN THIS
def train_model(model, train_loader, epochs, lr):
    """Train the model with accuracy and progress tracking."""
    try:
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
        
        # Loop for training epochs
        for epoch in range(epochs):
            start_time = time.time()
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            total_batches = len(train_loader)

            # For accurate tracking of progress, eta, loss, accuracy, etc.
            with tqdm(total=total_batches, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    optimizer.zero_grad()
                    outputs = model(inputs)

                    # I just needed to transform the data to 100x100 because for some reason it was 104x104 going into a 100x100
                    outputs_resized = F.interpolate(outputs, size=(100, 100), mode='bilinear', align_corners=False)
                    targets_resized = F.interpolate(targets, size=(100, 100), mode='bilinear', align_corners=False)
                    
                    # Calculate loss
                    loss = criterion(outputs_resized, targets_resized)
                    epoch_loss += loss.item()
                    
                    # Calculate accuracy
                    accuracy = calculate_accuracy(outputs_resized, targets_resized)
                    epoch_accuracy += accuracy
                    
                    # Backpropagation
                    loss.backward()
                    optimizer.step()
                    
                    # Update progress bar
                    pbar.set_postfix({'Loss': loss.item(), 'Accuracy': accuracy})
                    pbar.update(1)

            # Average loss and accuracy for the epoch
            avg_loss = epoch_loss / total_batches
            avg_accuracy = epoch_accuracy / total_batches
            end_time = time.time()
            epoch_duration = end_time - start_time
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.2f}%, Time: {epoch_duration:.2f}s')
        
        # Returns model after epoch loop    
        return model
    
    # Error handling
    except Exception as e:
        print(f"Error during training: {e}")
        return None
    
def save_model(model, save_folder='./trained_model/'):
    """Save the trained model."""
    try:
        # Makes the folder if it doesn't exist
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        # Adds model to the model path
        model_path = os.path.join(save_folder, 'colorization_model2.pth')
        torch.save(model.state_dict(), model_path)
        print(f'Model saved at {model_path}')

    # Error handling
    except Exception as e:
        print(f"Error saving the model: {e}")

def evaluate_model(model, test_loader, validation_loader, threshold=0.1):
    """Evaluate the model, print accuracy, and update progress bar."""
    try:
        model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0
        criterion = torch.nn.MSELoss()

        # Initialize tqdm progress bar
        progress_bar = tqdm(test_loader, desc="Evaluating", unit="batch")

        with torch.no_grad():
            for inputs, targets in progress_bar:
                outputs = model(inputs)
                
                # Resize outputs and targets to match dimensions
                outputs_resized = F.interpolate(outputs, size=(100, 100), mode='bilinear', align_corners=False)
                targets_resized = F.interpolate(targets, size=(100, 100), mode='bilinear', align_corners=False)
                
                # Calculate loss
                loss = criterion(outputs_resized, targets_resized).item()
                total_loss += loss

                # Calculate accuracy using the calculate_accuracy function
                accuracy = calculate_accuracy(outputs_resized, targets_resized, threshold)
                total_accuracy += accuracy
                total_samples += 1

                # Update progress bar
                progress_bar.set_postfix(loss=loss, accuracy=accuracy)

        # Validation step
        val_loss, val_accuracy = validate_model(model, validation_loader)
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        # Final loss and accuracy
        average_loss = total_loss / len(test_loader)
        average_accuracy = total_accuracy / total_samples
        print(f'Evaluation Loss: {average_loss:.4f}, Average Accuracy: {average_accuracy:.2f}%')

    # Error Handling
    except Exception as e:
        print(f"Error during evaluation: {e}")

def validate_model(model, val_loader, threshold=0.1):
    """Validate the model on the validation set."""
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    criterion = torch.nn.MSELoss()
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            outputs_resized = F.interpolate(outputs, size=(100, 100), mode='bilinear', align_corners=False)
            targets_resized = F.interpolate(targets, size=(100, 100), mode='bilinear', align_corners=False)
            
            loss = criterion(outputs_resized, targets_resized).item()
            total_loss += loss

            accuracy = calculate_accuracy(outputs_resized, targets_resized, threshold)
            total_accuracy += accuracy

    average_loss = total_loss / len(val_loader)
    average_accuracy = total_accuracy / len(val_loader)
    
    return average_loss, average_accuracy

def main():
    # Paths to the folders
    folder_path1 = './assets/images/bw/'
    folder_path2 = './assets/images/color/'

    # Load and preprocess images
    X, Y = load_and_preprocess_images(folder_path1, folder_path2)
    if X is None or Y is None:
        return
    # Debugging
    print(f'X shape: {X.shape}, Y shape: {Y.shape}')


    # Prepare data loaders
    train_loader, validation_loader, test_loader = prepare_data_loaders(X, Y, batch_size=32)
    if (train_loader is None) or (test_loader is None) or (validation_loader is None):
        return

    # Initialize the model
    model = ColorizationModel().to(device)

    # Train the model
    model = train_model(model, train_loader, epochs=1, lr=0.001)
    if model is None:
        return

    # Save the trained model
    save_model(model)

    # Evaluate the model
    evaluate_model(model, test_loader, validation_loader)

if __name__ == "__main__":
    main()