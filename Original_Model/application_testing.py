import os
import torch
import cv2 as cv2
import numpy as np
import torch.nn.functional as F
from PIL import Image
from skimage import color
from Original_Model.model import ColorizationModel  # Import the model

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loads model from the local model.pth
def load_model(model_path):
    model = ColorizationModel().to(device)
    model.load_state_dict(torch.load(model_path,weights_only=True))
    model.eval()
    return model

# Opens camera view and returns a gray image
def cam_maybe():
    # Camera port = 0 means the integrated webcam of the computer
    cam_port = 0
    # Just makes a video capture from the webcam
    cam = cv2.VideoCapture(cam_port)

    # In case camera doesn't open :)
    if cam.isOpened(): # try to get the first frame
        rval, frame = cam.read()
    else:
        rval = False

    while rval:
        cv2.imshow("Preview", frame)
        rval, frame = cam.read()
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            result, image = cam.read()
            if result:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                cv2.imshow("Grayscale",rgb_image)
                cv2.waitKey(0)
                cv2.destroyWindow("Grayscale")
                cv2.destroyWindow("Preview")
                cam.release()
            return rgb_image
        
def preprocess_input_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((100, 100))
    img = np.array(img) / 255.0
    img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().to(device)
    return img_tensor

def postprocess_output(output_array, input_tensor):
    # Initialize the LAB image array
    output_lab = np.zeros((100, 100, 3))
    
    # Set the L channel; assume input_tensor is in range [0, 1]
    output_lab[:, :, 0] = input_tensor.squeeze(0).squeeze(0).cpu().numpy() * 100
    
    # Set the A and B channels; assume output_array is normalized to [0, 1]
    output_lab[:, :, 1] = (output_array[:, :, 0] * 255) - 128
    output_lab[:, :, 2] = (output_array[:, :, 1] * 255) - 128
    
    # Convert LAB to RGB
    output_rgb = color.lab2rgb(output_lab)
    return output_rgb


def main():
    # Saves the gray image in a pathway
    gray_img = cam_maybe()
    path = './assets/application/original'
    cv2.imwrite(os.path.join(path , 'lmaoooooooooooooooooooooooooooooooooooooo.jpg'), gray_img)

    # Load the trained model
    model = ColorizationModel().to(device)
    model.load_state_dict(torch.load('./trained_model/colorization_model3.pth',weights_only=True))
    model.eval()

    # Preprocess the input image
    input_image_path = './assets/application/original/lmaoooooooooooooooooooooooooooooooooooooo.jpg'
    input_tensor = preprocess_input_image(input_image_path)

    # Get the model's output
    with torch.no_grad():
        output_tensor = model(input_tensor) 
        output_tensor = F.interpolate(output_tensor, size=(100, 100), mode='bilinear', align_corners=False)
        output_array = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Convert to RGB and display
    output_rgb = postprocess_output(output_array, input_tensor)
    # Normalize the output image to [0, 255] range and convert to uint8
    output_rgb = np.clip(output_rgb * 255, 0, 255).astype(np.uint8)
    output_rgb = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL) # Makes the window scalable by nature
    cv2.imshow("Output",output_rgb)
    cv2.waitKey(0)
    cv2.destroyWindow("Output")

if __name__ == "__main__":
    main()