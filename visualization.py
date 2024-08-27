import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.color import lab2rgb, rgb2gray

def visualize_prediction(img, output):
    output = output.squeeze(0).cpu().numpy()
    output = cv2.resize(output.transpose(1, 2, 0), (img.shape[1], img.shape[0]))

    # Convert LAB to RGB and display
    outputLAB = np.zeros((img.shape[0], img.shape[1], 3))
    outputLAB[:, :, 0] = rgb2gray(img)
    outputLAB[:, :, 1:] = output
    outputLAB = (outputLAB * [100, 255, 255]) - [0, 128, 128]
    rgb_image = lab2rgb(outputLAB)

    plt.imshow(rgb_image)
    plt.show()
