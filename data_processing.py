import os
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, rgb2gray
from skimage import color
from tqdm import tqdm

def load_images_from_folder(folder_path, name):
    images = []
    # Count the number of files in the folder for tqdm progress bar
    total_files = sum(len(files) for _, _, files in os.walk(folder_path))
    
    # Initialize tqdm progress bar
    progress_bar = tqdm(total=total_files, desc=f"Loading {name} Images", unit="image")
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Construct the full path to the image
            img_path = os.path.join(root, file)
            try:
                # Attempt to open the image
                img = Image.open(img_path).resize((100, 100))
                img = np.array(img) / 255.0
                images.append(img)
            except (OSError, ValueError) as e:
                # Handle exceptions for image loading or processing
                print(f"Error loading image {img_path}: {e}")
                continue

            # Update progress bar
            progress_bar.update(1)

    # Close the progress bar after processing
    progress_bar.close()
    return images

def preprocess_images(grayscale_images, color_images):
    gray_images = []
    processed_color_images = []

    # Initialize tqdm progress bar
    progress_bar = tqdm(total=len(grayscale_images), desc="Preprocessing Images", unit="image")
    for gray_img, color_img in zip(grayscale_images, color_images):
        try:
            # Ensure the grayscale image is already in the correct format
            gray_images.append(gray_img)

            # Convert color image to LAB color space and normalize
            if color_img.shape[-1] == 3:  # Ensure it's an RGB image
                lab_image = rgb2lab(color_img)
                lab_image_norm = (lab_image + [0, 128, 128]) / [100, 255, 255]
                processed_color_images.append(lab_image_norm[:, :, 1:])
            else:
                print(f"Skipping non-RGB color image with shape: {color_img.shape}")
        except Exception as e:
            # Print error message and continue with the next image
            print(f"Error processing image: {e}")
            continue
        
        # Update progress bar
        progress_bar.update(1)

    # Close the progress bar after processing
    progress_bar.close()
    return np.array(gray_images), np.array(processed_color_images)
