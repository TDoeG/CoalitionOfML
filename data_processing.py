import os
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, rgb2gray
from skimage import color

# Function to load and preprocess images from a specified folder
def load_images_from_folder(folder_path):
    images = []  # Initialize a list to store images
    for root, dirs, files in os.walk(folder_path):  # Traverse through the folder
        for file in files:  # Loop through each file in the folder
            img_path = os.path.join(root, file)  # Get the full path of the image file
            img = Image.open(img_path).resize((100, 100))  # Open and resize the image to 100x100 pixels
            img = np.array(img) / 255.0  # Convert the image to a numpy array and normalize pixel values to [0, 1]
            images.append(img)  # Append the processed image to the list
    return images  # Return the list of images

# Function to preprocess images for model training
def preprocess_images(images):
    # Convert images to grayscale
    gray_images = [rgb2gray(img) for img in images]  # Convert each image to grayscale using rgb2gray
    
    color_images = []  # Initialize a list to store the LAB color components
    for img in images:  # Loop through each image
        lab_image = rgb2lab(img)  # Convert the RGB image to LAB color space
        # Normalize the LAB image by shifting and scaling the L, A, and B channels
        lab_image_norm = (lab_image + [0, 128, 128]) / [100, 255, 255]
        # Extract only the A and B channels (discard the L channel) and add to the color_images list
        color_images.append(lab_image_norm[:, :, 1:])
    
    # Return the grayscale images and the normalized color images (A and B channels)
    return np.array(gray_images), np.array(color_images)
