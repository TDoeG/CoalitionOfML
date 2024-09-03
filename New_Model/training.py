import torch
import torch.nn as nn
import torch.optim as optim
from model import ConvNet

def calculate_accuracy(predictions, targets):
    # Calculate the Euclidean distance between the predicted color and the ground truth color
    euclidean_distance = torch.sqrt(torch.sum((predictions - targets) ** 2, dim=1))
    
    # Normalize the distance (smaller distance means higher accuracy)
    max_distance = torch.sqrt(torch.tensor(3.0))  # Max possible distance in RGB space
    accuracy = 1.0 - torch.mean(euclidean_distance / max_distance)
    
    return accuracy.item()

def train_model(x_train, y_train, x_test, y_test, batch_size, learning_rate, epochs):
    # Define model, criterion, and optimizer
    net = ConvNet(batch_size)
    net.cpu() # Change to cuda if NVIDIA
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # Training loop
    train_loss_container, test_loss_container = [], []
    train_accuracy_container, test_accuracy_container = [], []

    # Loop over the number of epochs
    for e in range(epochs):
        # Initialize the training and testing loss for the current epoch
        train_loss = 0.0
        test_loss = 0.0
        train_accuracy = 0.0
        test_accuracy = 0.0

        # Training phase
        # Loop over each batch of training data
        net.train()
        for batch, train_data in enumerate(x_train):
            # Extract the input and output (target) data for the current batch
            ip = train_data  # Input data (grayscale or first channel images)
            op = y_train[batch]  # Corresponding output data (color images)

            # Reset the gradients of the optimizer before backpropagation
            optimizer.zero_grad()

            # Forward pass: Compute the model's output
            model_op = net(ip, batch_size)

            # Calculate the loss between the model's output and the actual target
            loss = criterion(model_op, op)

            # Accumulate the training loss for the current batch
            train_loss += loss.item()

            # Backward pass: Compute the gradients
            loss.backward()

            # Update the model's parameters using the optimizer
            optimizer.step()

            # Calukate accuracy for the batch
            accuracy = calculate_accuracy(model_op, op)
            train_accuracy += accuracy

        # Testing phase (evaluation)
        # Disable gradient computation to speed up inference and reduce memory usage
        net.eval()
        with torch.no_grad():
            # Loop over each batch of testing data
            for batch_test, test_data in enumerate(x_test, 0):
                # Extract the input and output (target) data for the current batch
                ip_test = test_data  # Input data (grayscale or first channel images)
                op_test = y_test[batch_test]  # Corresponding output data (color images)

                # Forward pass: Compute the model's output
                model_op = net(ip_test, batch_size)

                # Calculate the loss between the model's output and the actual target
                loss_test = criterion(model_op, op_test)

                # Accumulate the testing loss for the current batch
                test_loss += loss_test.item()

                # Calculate accuracy for the batch
                accuracy_test = calculate_accuracy(model_op, op_test)
                test_accuracy += accuracy_test

        # Store the total training and testing loss for the current epoch in the containers
        train_loss_container.append(train_loss)
        test_loss_container.append(test_loss)
        train_accuracy_container.append(train_accuracy / len(x_train))
        test_accuracy_container.append(test_accuracy / len(x_test))

        # Print the loss for the current epoch
        print(f'EPOCH: {e + 1} | Train_loss: {train_loss:.6f} | Test_loss: {test_loss:.6f} | ' f'Train_acc: {train_accuracy / len(x_train):.4f} | Test_acc: {test_accuracy / len(x_test):.4f}')

    # Print a message to indicate that training has completed
    print('\nFinished Training')
    return net, train_loss_container, test_loss_container, train_accuracy_container, test_accuracy_container
