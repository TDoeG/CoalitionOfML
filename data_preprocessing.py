import os
from PIL import Image

def convert_images_to_greyscale(source_folder, destination_folder):
    # Loop through all the directories and files in the source folder
    for root, dirs, files in os.walk(source_folder):
        # Create corresponding subdirectories in the destination folder
        relative_path = os.path.relpath(root, source_folder)
        dest_subfolder = os.path.join(destination_folder, relative_path)
        
        # Create the destination subfolder if it doesn't exist
        if not os.path.exists(dest_subfolder):
            os.makedirs(dest_subfolder)
        
        # Loop through each file in the current directory
        for filename in files:
            # Ensure that filename is a string and then check if it ends with a valid image extension
            if isinstance(filename, str) and filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')):
                # Construct full file path
                img_path = os.path.join(root, filename)
                
                # Open the image
                img = Image.open(img_path)
                
                # Convert the image to grayscale
                grayscale_img = img.convert('L')
                
                # Construct the full path for the destination image
                save_path = os.path.join(dest_subfolder, filename)
                
                # Save the grayscale image to the destination folder
                grayscale_img.save(save_path)
                print(f"Saved grayscale image: {save_path}")

# Example usage
source_folder = './assets/images/'  # Replace with the path to your source folder
destination_folder = './assets/images/results/'  # Replace with the path to your destination folder

convert_images_to_greyscale(source_folder, destination_folder)

