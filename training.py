import torch
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, random_split
from model import ColorizationModel  # Import the model
from data_processing import load_images_from_folder, preprocess_images  # Import data processing functions
from visualization import visualize_prediction  # Import visualization function
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_preprocess_images(grayscale_path, colorized_path):
    """Load and preprocess images."""
    try:
        grayscale = load_images_from_folder(grayscale_path)
        colorized = load_images_from_folder(colorized_path)
        X, Y = preprocess_images(grayscale, colorized)
        return X, Y
    except Exception as e:
        print(f"Error loading or preprocessing images: {e}")
        return None, None

def prepare_data_loaders(X, Y, batch_size=32):
    """Convert data to tensors, split into train/test sets, and create data loaders."""
    try:
        X_tensor = torch.tensor(X).unsqueeze(1).float().to(device)
        Y_tensor = torch.tensor(Y).permute(0, 3, 1, 2).float().to(device)
        
        # Split the dataset
        dataset_size = len(X_tensor)
        train_size = int(0.8 * dataset_size)
        test_size = dataset_size - train_size
        train_X, test_X = random_split(X_tensor, [train_size, test_size])
        train_Y, test_Y = random_split(Y_tensor, [train_size, test_size])
        
        # Stack tensors
        train_X = torch.stack([train_X[i] for i in range(len(train_X))])
        train_Y = torch.stack([train_Y[i] for i in range(len(train_Y))])
        test_X = torch.stack([test_X[i] for i in range(len(test_X))])
        test_Y = torch.stack([test_Y[i] for i in range(len(test_Y))])
        
        # Create DataLoader
        train_dataset = TensorDataset(train_X, train_Y)
        test_dataset = TensorDataset(test_X, test_Y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        return train_loader, test_loader
    except Exception as e:
        print(f"Error preparing data loaders: {e}")
        return None, None

def train_model(model, train_loader, epochs=100, lr=0.001):
    """Train the model."""
    try:
        criterion = torch.nn.MSELoss()
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')
        
        return model
    except Exception as e:
        print(f"Error during training: {e}")
        return None

def save_model(model, save_folder='./trained_model/'):
    """Save the trained model."""
    try:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        model_path = os.path.join(save_folder, 'colorization_model.pth')
        torch.save(model.state_dict(), model_path)
        print(f'Model saved at {model_path}')
    except Exception as e:
        print(f"Error saving the model: {e}")

def evaluate_model(model, test_loader):
    """Evaluate the model."""
    try:
        model.eval()
        loss = 0.0
        criterion = torch.nn.MSELoss()
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss += criterion(outputs, targets).item()
        print(f'Evaluation Loss: {loss / len(test_loader)}')
    except Exception as e:
        print(f"Error during evaluation: {e}")

def main():
    # Paths to the folders
    folder_path1 = './assets/images/bw/'
    folder_path2 = './assets/images/color/'

    # Load and preprocess images
    X, Y = load_and_preprocess_images(folder_path1, folder_path2)
    if X is None or Y is None:
        return  # Exit if data loading or preprocessing fails

    # Prepare data loaders
    train_loader, test_loader = prepare_data_loaders(X, Y, batch_size=32)
    if train_loader is None or test_loader is None:
        return  # Exit if data loader preparation fails

    # Initialize the model
    model = ColorizationModel().to(device)

    # Train the model
    model = train_model(model, train_loader, epochs=100, lr=0.001)
    if model is None:
        return  # Exit if training fails

    # Save the trained model
    save_model(model)

    # Evaluate the model
    evaluate_model(model, test_loader)

    # (Optional) Prediction and visualization example
    # Uncomment the lines below to visualize predictions
    # img = Image.open('path_to_image').resize((100, 100))
    # img = np.array(img) / 255.0
    # X = torch.tensor(color.rgb2gray(img)).unsqueeze(0).unsqueeze(0).float().to(device)
    #
    # with torch.no_grad():
    #     output = model(X)
    #
    # visualize_prediction(img, output)

if __name__ == "__main__":
    main()