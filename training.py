import torch
import torch.optim as optim
import numpy as np
import os
from skimage import color
from torch.utils.data import DataLoader, TensorDataset
from model import ColorizationModel  # Import the model
from data_processing import load_images_from_folder, preprocess_images  # Import data processing functions
from visualization import visualize_prediction  # Import visualization function
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess images
folder_path1 = './assets/images/bw/'
folder_path2 = './assets/images/color/'
grayscale = load_images_from_folder(folder_path1)
colorized = load_images_from_folder(folder_path2)

X, Y = preprocess_images(grayscale,colorized)

# Convert to PyTorch tensors
X = torch.tensor(X).unsqueeze(1).float().to(device)
Y = torch.tensor(Y).permute(0, 3, 1, 2).float().to(device)

# Create DataLoader
dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)

# Split dataset into training and testing sets (80% training, 20% testing)
train_size = int(0.8 * len(X))
test_size = len(X) - train_size
train_X, test_X = torch.utils.data.random_split(X, [train_size, test_size])
train_Y, test_Y = torch.utils.data.random_split(Y, [train_size, test_size])

# Create DataLoader for training and testing sets
train_dataset = TensorDataset(train_X, train_Y)
test_dataset = TensorDataset(test_X, test_Y)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)

# Initialize the model
model = ColorizationModel().to(device)

# Define loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Save the trained model
save_folder = './trained_model/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

model_path = os.path.join(save_folder, 'colorization_model.pth')
torch.save(model.state_dict(), model_path)
print(f'Model saved at {model_path}')

# Evaluate the model
model.eval()
with torch.no_grad():
    loss = 0.0
    for inputs, targets in dataloader:
        outputs = model(inputs)
        loss += criterion(outputs, targets).item()
    print(f'Evaluation Loss: {loss / len(dataloader)}')

# Prediction and visualization
#img = Image.open('path_to_image').resize((100, 100))
#img = np.array(img) / 255.0
#X = torch.tensor(color.rgb2gray(img)).unsqueeze(0).unsqueeze(0).float().to(device)
#
#with torch.no_grad():
#    output = model(X)

#visualize_prediction(img, output)
