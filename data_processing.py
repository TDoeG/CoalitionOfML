import os
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, rgb2gray
from skimage import color

def load_images_from_folder(folder_path):
    images = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            img_path = os.path.join(root, file)
            img = Image.open(img_path).resize((100, 100))
            img = np.array(img) / 255.0
            images.append(img)
    return images

def preprocess_images(images):
    gray_images = [rgb2gray(img) for img in images]
    color_images = []
    for img in images:
        lab_image = rgb2lab(img)
        lab_image_norm = (lab_image + [0, 128, 128]) / [100, 255, 255]
        color_images.append(lab_image_norm[:, :, 1:])
    return np.array(gray_images), np.array(color_images)
