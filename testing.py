import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import ColorizationModel  # Import the model
from training import preprocess_images, load_images_from_folder   # Import data processing functions
from training import evaluate_model

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    """Load the saved model from the .pth file."""
    model = ColorizationModel().to(device)
    model.load_state_dict(torch.load(model_path,weights_only=True))
    model.eval()
    return model

def main():
    # Load the saved model
    model_path = './trained_model/colorization_model1.pth'  # Adjust path as needed
    model = load_model(model_path)

    # Load and preprocess the test data
    folder_path1 = './assets/images/bw/'  # Path to grayscale images
    folder_path2 = './assets/images/color/'  # Path to color images
    grayscale = load_images_from_folder(folder_path1)
    colorized = load_images_from_folder(folder_path2)
    
    X_test, Y_test = preprocess_images(grayscale, colorized)
    X_test_tensor = torch.tensor(X_test).unsqueeze(1).float()
    Y_test_tensor = torch.tensor(Y_test).permute(0, 3, 1, 2).float()

    # Create test DataLoader
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, Y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Evaluate the model
    evaluate_model(model, test_loader)

if __name__ == '__main__':
    main()
