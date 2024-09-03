import numpy as np
import torch
import cv2
from model import ConvNet

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model from the local model.pth
def load_model(model_path, batch_size):
    model = ConvNet(batch_size)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model

def open_camera(model):
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Error: Camera not accessible")
        return

    while True:
        rval, frame = cam.read()
        if not rval:
            print("Error: Failed to grab frame")
            break

        cv2.imshow("Preview In Color", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC key
            cv2.destroyWindow("Preview In Color")
            new_image = cv2.resize(frame, (32, 32))
            image, grayscale_img = processImage(model, new_image, batch_size=100)
            
            # Creates grayscale window
            cv2.namedWindow("Grayscale", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Grayscale", 1280, 720)
            cv2.imshow("Grayscale", grayscale_img)
            cv2.waitKey(0)
            cv2.destroyWindow("Grayscale")

            # Creates color image
            cv2.namedWindow("Predicted Color Image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Predicted Color Image", 1280, 720)
            cv2.imshow("Predicted Color Image", image)
            cv2.waitKey(0)
            cv2.destroyWindow("Predicted Color Image")
        
        elif key == ord('q'):  # Press 'q' to quit
            break

    cam.release()
    cv2.destroyAllWindows()

def processImage(model, img, batch_size):
    # Convert the image from RGB to grayscale using OpenCV
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # OpenCV reads images in BGR

    # Ensure the image size matches the expected input size of 32x32
    if gray_img.shape[:2] != (32, 32):
        gray_img = cv2.resize(gray_img, (32, 32))

    # Expand dimensions to match the model's expected input shape [batch_size, channels, height, width]
    ip_data = np.expand_dims(gray_img, axis=0)  # Add channel dimension
    ip_data = np.expand_dims(ip_data, axis=0)  # Add batch dimension

    # Convert to a PyTorch tensor and move to device
    data = torch.from_numpy(ip_data).float().to(device)

    # Normalize the input tensor
    mean = torch.mean(data)
    std = torch.std(data)
    data = (data - mean) / std

    # Get the model prediction for the image
    with torch.no_grad():
        output = model(data, batch_size)
    
    # Gets output and reshapes np array and converts from normalized into color values
    tmp = output.detach().cpu().numpy().reshape(32,32,3) * 255

    # Ensure values are in the valid range [0, 255]
    output2 = np.clip(tmp, 0, 255).astype(np.uint8)
    output_bgr = cv2.cvtColor(output2, cv2.COLOR_RGB2BGR)

    return output_bgr, gray_img

def main():
    batch_size = 1
    model_path = './New_Model/saved_model/model(2).pth'
    model = load_model(model_path, batch_size)
    open_camera(model)

if __name__ == "__main__":
    main()
