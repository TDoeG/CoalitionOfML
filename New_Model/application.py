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
    model.load_state_dict(torch.load(model_path,weights_only=True))
    model.eval()
    return model

def openCameraMaybe(model,batch_size):
    cam_port = 0
    cam = cv2.VideoCapture(cam_port)

    if cam.isOpened():
        rval, frame = cam.read()
    else:
        rval = False

    while rval:
        cv2.imshow("Preview In Color", frame)
        rval, frame = cam.read()
        key = cv2.waitKey(0)
        if key == 27:
            result, image = cam.read()
            if result:
                cv2.destroyWindow("Preview In Color")
                new_image = image.resize(32,32)
                image, grayscale_img = processImage(model,new_image,batch_size)
                cv2.imshow("Grayscale",grayscale_img)
                cv2.waitKey(0)
                cv2.destroyWindow("Grayscale")
                cv2.imshow("Predicted Color Image", image)
                cv2.waitKey(0)
                cv2.destroyWindow("Predicted Color Image")
                cv2.release()

def processImage(model,img,batch_size):
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
    return img, gray_img

def main():
    model_path = './saved_model/Model(1).pth'
    batch_size = 100
    model = load_model(model_path,batch_size)
    openCameraMaybe(model,batch_size)

if __name__ == "__main__":
    main()