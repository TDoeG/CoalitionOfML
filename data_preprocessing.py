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

def resize_images_in_folder(folder_path, target_size=(100, 100)):
    # Loop through all the directories and files in the folder
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            # Ensure that filename is a string and ends with a valid image extension
            if isinstance(filename, str) and filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.JPG')):
                img_path = os.path.join(root, filename)  # Construct full file path
                try:
                    # Attempt to open the image
                    img = Image.open(img_path)
                    
                    # Resize the image to the target size
                    resized_img = img.resize(target_size)
                    
                    # Save the resized image, replacing the original one
                    resized_img.save(img_path)
                    print(f"Resized and saved image: {img_path}")
                    
                    # Opens new image
                    img = Image.open(img_path)

                    # If new image is not size 100x100, attempt to delete
                    if img.size[0] != 100 or img.size[1] != 100:
                        try:
                            os.remove(img_path)
                            print(f"Deleted corrupted or problematic image: {img_path}")
                        except Exception as delete_error:
                            print(f"Failed to delete image {img_path}: {delete_error}")
                except Exception as e:
                    # Print the error message
                    print(f"Error processing file {img_path}: {e}")
                    
                    # Attempt to delete the error image
                    try:
                        os.remove(img_path)
                        print(f"Deleted corrupted or problematic image: {img_path}")
                    except Exception as delete_error:
                        print(f"Failed to delete image {img_path}: {delete_error}")



                    
source_folder = './assets/images/color/'  # Replace with the path to your source folder
destination_folder = './assets/images/bw/'  # Replace with the path to your destination folder

convert_images_to_greyscale(source_folder, destination_folder)
resize_images_in_folder(source_folder)
resize_images_in_folder(destination_folder)