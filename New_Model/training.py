import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from model import ConvNet

def train_model(x_train, y_train, x_test, y_test, batch_size, learning_rate, epochs):
    # Define model, criterion, and optimizer
    net = ConvNet(batch_size)
    net.cpu() # Change to cuda if NVIDIA
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # Training loop
    train_loss_container, test_loss_container = [], []

    # Loop over the number of epochs
    for e in range(epochs):
        # Initialize the training and testing loss for the current epoch
        train_loss = 0.0
        test_loss = 0.0

        # Training phase
        # Loop over each batch of training data
        for batch, train_data in enumerate(x_train):
            # Extract the input and output (target) data for the current batch
            ip = train_data  # Input data (grayscale or first channel images)
            op = y_train[batch]  # Corresponding output data (color images)

            # Reset the gradients of the optimizer before backpropagation
            optimizer.zero_grad()

            # Forward pass: Compute the model's output
            model_op = net(ip)

            # Calculate the loss between the model's output and the actual target
            loss = criterion(model_op, op)

            # Accumulate the training loss for the current batch
            train_loss += loss.item()

            # Backward pass: Compute the gradients
            loss.backward()

            # Update the model's parameters using the optimizer
            optimizer.step()

        # Testing phase (evaluation)
        # Disable gradient computation to speed up inference and reduce memory usage
        with torch.no_grad():
            # Loop over each batch of testing data
            for batch_test, test_data in enumerate(x_test, 0):
                # Extract the input and output (target) data for the current batch
                ip_test = test_data  # Input data (grayscale or first channel images)
                op_test = y_test[batch_test]  # Corresponding output data (color images)

                # Forward pass: Compute the model's output
                model_op = net(ip_test)

                # Calculate the loss between the model's output and the actual target
                loss_test = criterion(model_op, op_test)

                # Accumulate the testing loss for the current batch
                test_loss += loss_test.item()

        # Store the total training and testing loss for the current epoch in the containers
        train_loss_container.append(train_loss)
        test_loss_container.append(test_loss)

        # Print the loss for the current epoch
        print('\rEPOCH: {} | Train_loss: {:.6f} | Test_loss: {:.6f}'.format(e + 1, train_loss, test_loss), end='')

    # Print a message to indicate that training has completed
    print('\nFinished Training')
    return net, train_loss_container, test_loss_container
