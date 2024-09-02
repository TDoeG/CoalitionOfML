# Application (Camera)
import numpy as np
import torch
import cv2
from model import ConvNet

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loads model from the local model.pth
def load_model(model_path,batch_size):
    model = ConvNet(batch_size).to(device)
    model.load_state_dict(torch.load(model_path,map_location=device,weights_only=True))
    model.eval()
    return model

def openCameraMaybe(model,batch_size):
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
            image, grayscale_img = processImage(model, new_image, batch_size)
            cv2.imshow("Grayscale", grayscale_img)
            cv2.waitKey(0)
            cv2.destroyWindow("Grayscale")
            cv2.imshow("Predicted Color Image", image)
            cv2.waitKey(0)
            cv2.destroyWindow("Predicted Color Image")
        
        elif key == ord('q'):  # Press 'q' to quit
            break

    cam.release()
    cv2.destroyAllWindows()

"""def processImage(model,img,batch_size):
    grayscale_im = []
    # Convert the image from RGB to grayscale using OpenCV
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    grayscale_im.append(gray_img)
    ip_data = np.array(grayscale_im)
    
    
    data = torch.from_numpy(ip_data).float()
    data2 = data.contiguous().view(data.size(0),batch_size,-1,32,32)
    mean = torch.mean(data2[:, :, 0, :, :])
    std = torch.std(data2[:, :, 0, :, :])
    data2[:, :, 0, :, :] = (data2[:, :, 0, :, :] - mean) / std
    
    # Get the model prediction for the image
    tmp = model(data2).detach().cpu().numpy().reshape(32, 32, 3) * 255
    img = tmp.astype(np.uint8)
    return img, gray_img"""

"""def processImage(model, img, batch_size):
    # Convert the image from RGB to grayscale using OpenCV
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Use BGR since OpenCV reads images in BGR

    # Ensure the image size matches the expected input size of 32x32
    if gray_img.shape[:2] != (32, 32):
        gray_img = cv2.resize(gray_img, (32, 32))

    # Expand dimensions to match the model's expected input shape [batch_size, channels, height, width]
    ip_data = np.expand_dims(gray_img, axis=0)  # Add channel dimension
    ip_data = np.expand_dims(ip_data, axis=0)  # Add batch dimension

    # Convert to a PyTorch tensor and move to device
    data = torch.from_numpy(ip_data).float().to(device)

    # The model expects a batch size of 100, so you should adjust the input tensor accordingly
    if data.size(0) != batch_size:
        # Create a batch of size `batch_size` from the single image
        data2 = data.repeat(batch_size, 1, 1, 1)
    else:
        data2 = data

    # Normalize the input tensor (Ensure the mean and std are the same as used in training)
    mean = torch.mean(data2)
    std = torch.std(data2)
    data2 = (data2 - mean) / std

    # Get the model prediction for the image
    with torch.no_grad():
        output = model(data2)

    # Post-process the output
    output = output.squeeze(0)  # Remove the batch dimension
    output = output.permute(1, 2, 0).cpu().numpy() * 255  # Convert to shape [32, 32, 3]
    img = output.astype(np.uint8)

    return img, gray_img"""

def processImage(model, img, batch_size):
    # Convert the image from RGB to grayscale using OpenCV
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Use BGR since OpenCV reads images in BGR

    # Ensure the image size matches the expected input size of 32x32
    if gray_img.shape[:2] != (32, 32):
        gray_img = cv2.resize(gray_img, (32, 32))

    # Expand dimensions to match the model's expected input shape [batch_size, channels, height, width]
    ip_data = np.expand_dims(gray_img, axis=0)  # Add channel dimension
    ip_data = np.expand_dims(ip_data, axis=0)  # Add batch dimension

    # Convert to a PyTorch tensor and move to device
    data = torch.from_numpy(ip_data).float().to(device)

    # The model expects a batch size of 100, so you should adjust the input tensor accordingly
    if data.size(0) != batch_size:
        # Create a batch of size `batch_size` from the single image
        data2 = data.repeat(batch_size, 1, 1, 1)
    else:
        data2 = data

    # Normalize the input tensor (Ensure the mean and std are the same as used in training)
    mean = torch.mean(data2)
    std = torch.std(data2)
    data2 = (data2 - mean) / std

    # Get the model prediction for the image
    with torch.no_grad():
        output = model(data2)
    
    # Check the shape of the output tensor
    print(f"Output shape: {output.shape}")

    # Post-process the output
    output = output.squeeze(0)  # Remove the batch dimension
    
    # If output has 4 dimensions [batch_size, channels, height, width]
    if output.dim() == 4 or output.dim() == 3:
        output = output.permute(1, 2, 0).cpu().numpy() * 255  # Convert to shape [height, width, channels]
    else:
        raise RuntimeError(f"Unexpected output dimensions: {output.dim()}")

    img = output.astype(np.uint8)
    
    return img, gray_img

def main():
    model_path = './New_Model/saved_model/model(2).pth'
    batch_size = 100
    model = load_model(model_path,batch_size)
    openCameraMaybe(model,batch_size)

if __name__ == "__main__":
    main()