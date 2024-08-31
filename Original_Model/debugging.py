from PIL import Image

def check_image_mode(image_path):
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Get the image mode
        mode = img.mode
        
        # Print or return the mode
        print(f"The image mode is: {mode}")
        
        # Check if the mode is 'LAB'
        if mode == 'LAB':
            print("The image is in LAB format.")
        else:
            print("The image is not in LAB format.")
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")

# Example usage
image_path = 'C:/Users/tyler/OneDrive/Documents/Github/GrayscaleToColorizedML/assets/images/bw/1.jpg'
check_image_mode(image_path)
