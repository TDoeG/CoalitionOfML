import cv2
import torch
import numpy as np
from skimage import color
from model import ColorizationModel  # Import your model
from PIL import Image

# Function to load the model
def load_model(model_path, device):
    model = ColorizationModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set the model to evaluation mode
    return model

# Function to preprocess the grayscale image
def preprocess_image(image):
    # Convert to numpy array and normalize to [0, 1]
    image = np.array(image) / 255.0
    # Add a channel dimension to match the input shape of the model
    L = image[np.newaxis, np.newaxis, :, :]
    return torch.tensor(L).float()

# Function to postprocess the output
def postprocess_output(L_channel, AB_tensor):
    # Convert tensors to numpy arrays
    AB_channels = AB_tensor.squeeze().cpu().numpy()
    L_channel = L_channel.squeeze().cpu().numpy()

    # Stack L and AB channels to form the LAB image
    LAB_image = np.stack([L_channel, AB_channels[0], AB_channels[1]], axis=2)

    # Convert LAB to BGR for display
    BGR_image = color.lab2rgb(LAB_image)  # Convert to [0, 1] range
    BGR_image = (BGR_image * 255).astype(np.uint8)  # Convert to [0, 255] range for OpenCV

    return BGR_image

# Function to capture an image from the camera, process it, and display it
def capture_and_colorize(model, device):
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert the frame to grayscale (L format)
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayscale_image = Image.fromarray(grayscale_frame).convert("L")  # Convert to L format

        # Preprocess the grayscale image
        L_tensor = preprocess_image(grayscale_image).to(device)

        # Forward pass through the model to get the AB channels
        with torch.no_grad():
            AB_output = model(L_tensor)  # Predict the AB channels

        # Postprocess the output to form a BGR image
        colorized_image = postprocess_output(L_tensor, AB_output)

        # Display the resulting BGR image
        cv2.imshow('Colorized Image', colorized_image)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Load the saved model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = './trained_model/colorization_model1.pth'
    model = load_model(model_path, device)

    # Capture and colorize the image
    capture_and_colorize(model, device)

if __name__ == "__main__":
    main()
