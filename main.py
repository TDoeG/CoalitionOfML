import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from skimage.color import lab2rgb, rgb2lab
from skimage import color
# import glob
import cv2 as cv2
import os
import matplotlib.pyplot as plt
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model
class ColorizationModel(nn.Module):
    def __init__(self):
        super(ColorizationModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Load images
def load_images_from_folder(folder_path):
    images = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            img_path = os.path.join(root, file)
            img = Image.open(img_path).resize((100, 100))
            img = np.array(img) / 255.0
            images.append(img)
    return images

# Preprocess images
def preprocess_images(images):
    gray_images = [color.rgb2gray(img) for img in images]
    color_images = []
    for img in images:
        lab_image = rgb2lab(img)
        lab_image_norm = (lab_image + [0, 128, 128]) / [100, 255, 255]
        color_images.append(lab_image_norm[:, :, 1:])
    return np.array(gray_images), np.array(color_images)

folder_path1 = './assets/images/results/'
folder_path2 = './assets/images/'
images1 = load_images_from_folder(folder_path1)
images2 = load_images_from_folder(folder_path2)

X, Y = preprocess_images(images1)

# Convert to PyTorch tensors
X = torch.tensor(X).unsqueeze(1).float().to(device)
Y = torch.tensor(Y).permute(0, 3, 1, 2).float().to(device)

# Create DataLoader
dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

# Initialize the model
model = ColorizationModel().to(device)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.001)

# Training loop
epochs = 400
for epoch in range(epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Evaluate the model
model.eval()
with torch.no_grad():
    loss = 0.0
    for inputs, targets in dataloader:
        outputs = model(inputs)
        loss += criterion(outputs, targets).item()
    print(f'Evaluation Loss: {loss / len(dataloader)}')

# Prediction and visualization
img = Image.open('path_to_image').resize((100, 100))
img = np.array(img) / 255.0
X = torch.tensor(color.rgb2gray(img)).unsqueeze(0).unsqueeze(0).float().to(device)

with torch.no_grad():
    output = model(X)
output = output.squeeze(0).cpu().numpy()
output = cv2.resize(output.transpose(1, 2, 0), (img.shape[1], img.shape[0]))

# Convert LAB to RGB and display
outputLAB = np.zeros((img.shape[0], img.shape[1], 3))
outputLAB[:, :, 0] = color.rgb2gray(img)
outputLAB[:, :, 1:] = output
outputLAB = (outputLAB * [100, 255, 255]) - [0, 128, 128]
rgb_image = lab2rgb(outputLAB)

plt.imshow(rgb_image)
plt.show()
