import os
from PIL import Image, ImageFile

# Ensure that truncated images are loaded
ImageFile.LOAD_TRUNCATED_IMAGES = True

def convert_images_to_greyscale(source_folder, destination_folder):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Loop through all directories and files in the source folder
    for root, dirs, files in os.walk(source_folder):
        for filename in files:
            # Check if the file is an image
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')):
                try:
                    # Construct the full path to the image
                    img_path = os.path.join(root, filename)
                    
                    # Open the image
                    img = Image.open(img_path)
                    
                    # Convert the image to grayscale
                    grayscale_img = img.convert('L')
                    
                    # Construct the path for the destination image
                    save_path = os.path.join(destination_folder, filename)
                    
                    # Save the grayscale image to the destination folder
                    grayscale_img.save(save_path)
                    print(f"Saved grayscale image: {save_path}")
                
                except OSError as e:
                    # Print the error and skip the image
                    print(f"Error processing {img_path}: {e}")
                    
source_folder = './assets/images/color/'  # Replace with the path to your source folder
destination_folder = './assets/images/bw/'  # Replace with the path to your destination folder

convert_images_to_greyscale(source_folder, destination_folder)