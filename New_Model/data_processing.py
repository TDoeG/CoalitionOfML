import torch
import numpy as np
import torchvision.datasets as dataset_loader
import torchvision.transforms as transforms
import cv2
import random

random.seed(42)

def load_cifar10_dataset(root_dir, batch_size):
    # Load the CIFAR-10 dataset
    cifar_dataset = dataset_loader.CIFAR10(root=root_dir, train=True, download=True, transform=transforms.ToTensor())
    num_samples = len(cifar_dataset)
    return cifar_dataset, num_samples

def create_batches(cifar_dataset, num_samples, batch_size):
    # Create batches of data
    op_data = []
    ip_data = []

    for _ix in range(0, num_samples, batch_size):
        batch = [cifar_dataset[i][0] for i in range(_ix, min(_ix + batch_size, num_samples))]
        op_data.append(torch.stack(batch))  # Stack to create a batch tensor

        gray_batch = [transforms.Grayscale()(img) for img in batch]  # Convert to grayscale
        ip_data.append(torch.stack(gray_batch))  # Stack to create a batch tensor

    op_data = torch.cat(op_data)  # Combine batches into a single tensor
    ip_data = torch.cat(ip_data)  # Combine batches into a single tensor

    return ip_data, op_data

def split_data(ip_data, op_data, train_ratio=0.8):
    # Split the data into training and testing sets
    rand_ix = np.random.permutation(ip_data.shape[0])
    train_ix, test_ix = rand_ix[:int(train_ratio * rand_ix.shape[0])], rand_ix[int(train_ratio * rand_ix.shape[0]):]
    x_train, y_train = ip_data[train_ix], op_data[train_ix]
    x_test, y_test = ip_data[test_ix], op_data[test_ix]

    return x_train, y_train, x_test, y_test

def normalize_data(x_train, x_test, y_train, y_test):
    # Normalize the data
    mean = torch.mean(x_train)
    std = torch.std(x_train)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    y_train = y_train / 255
    y_test = y_test / 255

    return x_train, x_test, y_train, y_test
