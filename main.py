import numpy as np # linear algebra
from IPython.display import display, Image
from matplotlib.pyplot import imshow
from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, load_img
from skimage.color import lab2rgb, rgb2lab
from skimage import color
import keras as keras
import tensorflow as tf
import glob
import cv2 as cv2
import os

folder_path='./assets/images/results/' 
images1 = []
for img in os.walk(folder_path):
    img=folder_path+img
    img = load_img(img, target_size=(100,100)) 
    img = img_to_array(img)/ 255
    X= color.rgb2gray(img)
    images1.append(X)

folder_path='.assets/images/' 
images2 = []
for img in os.walk(folder_path):
    img=folder_path+img
    img = load_img(img, target_size=(100,100)) 
    img = img_to_array(img)/ 255
    lab_image = rgb2lab(img)
    lab_image_norm = (lab_image + [0, 128, 128]) / [100, 255, 255]
    # The input will be the black and white layer
    Y = lab_image_norm[:,:,1:]

    images2.append(Y)

# Creates an array of the corresponding images (Grey with Color)
X = np.array(images1)
Y = np.array(images2)

