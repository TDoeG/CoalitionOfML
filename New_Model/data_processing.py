import torch
import numpy as np
import torchvision.datasets as dataset_loader
import torchvision.transforms as transforms
import cv2
import random

random.seed(42)

def load_cifar10_dataset(root_dir):
    # Load the CIFAR-10 dataset
    cifar_dataset = dataset_loader.CIFAR10(root=root_dir, train=True, download=True, transform=transforms.ToTensor())
    cifar_dataset = cifar_dataset.data
    return cifar_dataset

def create_batches(cifar_dataset, batch_size):
    op_data = []
    for _ix in range(0,cifar_dataset.shape[0], batch_size):
        # Slice the dataset to get a batch of size 'batch_size'
        batch = cifar_dataset[_ix:_ix+batch_size]
        op_data.append(batch)
    op_data = np.array(op_data)

    ip_data = []
    grayscale_im = []
    for img in cifar_dataset:
        # Convert the image from RGB to grayscale using OpenCV
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        grayscale_im.append(gray_img)
    ip_data = np.array(grayscale_im)

    batches_grayscale_im = []
    for _ix in range(0, ip_data.shape[0], batch_size):
        # Slice the grayscale images array to get a batch of size 'batch_size'
        batch = ip_data[_ix:_ix + batch_size]
        batches_grayscale_im .append(batch)
    ip_data = np.array(batches_grayscale_im)
    
    return ip_data, op_data

def split_data(ip_data, op_data, train_ratio=0.8):
    # Split the data into training and testing sets
    test_ratio = 1-train_ratio

    # This will shuffle the order of the data to ensure randomness in the split
    rand_ix = np.random.permutation(ip_data.shape[0])

    # Calculate the number of samples that will be in the training set
    num_train_samples = int(train_ratio * rand_ix.shape[0])

    # The first 'num_train_samples' indices go to the training set
    train_ix = rand_ix[:num_train_samples]

    # The remaining indices go to the testing set
    test_ix = rand_ix[num_train_samples:]

    # For the training set, select the images and their corresponding labels using 'train_ix'
    x_train = ip_data[train_ix, :, :]  # Grayscale training images
    y_train = op_data[train_ix, :, :]  # Color images corresponding to the training images

    # For the testing set, select the images and their corresponding labels using 'test_ix'
    x_test = ip_data[test_ix, :, :]    # Grayscale testing images
    y_test = op_data[test_ix, :, :]    # Color images corresponding to the testing images

    return x_train, y_train, x_test, y_test

def transform_and_create_torch_tensors(data,batch_size):
    # Reshaping the input
    data = torch.from_numpy(data).float()
    return data.contiguous().view(data.size(0),batch_size,-1,32,32)

def normalize_data(x_train, x_test, y_train, y_test):
    # Calculate the mean of the first channel (assuming the data is in the shape (batch_size, channels, height, width))
    mean = torch.mean(x_train[:, :, 0, :, :])

    # Calculate the standard deviation of the first channel
    std = torch.std(x_train[:, :, 0, :, :])

    # Normalize the first channel of the training data
    x_train[:, :, 0, :, :] = (x_train[:, :, 0, :, :] - mean) / std

    # Normalize the first channel of the testing data using the mean and std from the training data
    x_test[:, :, 0, :, :] = (x_test[:, :, 0, :, :] - mean) / std

    # Normalize the output (y_train) by dividing by 255
    y_train = y_train / 255

    # Normalize the output (y_test) by dividing by 255
    y_test = y_test / 255

    return x_train, x_test, y_train, y_test, mean, std
