import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import ConvNet

def train_model(x_train, y_train, x_test, y_test, batch_size, learning_rate, epochs):
    # Define model, criterion, and optimizer
    net = ConvNet(batch_size)
    net.cpu()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # Training loop
    train_loss_container, test_loss_container = [], []
    train_acc_container, test_acc_container = [], []

    for e in tqdm(range(epochs), desc="Training Progress", unit="epoch"):
        train_loss = 0.0
        test_loss = 0.0
        correct_train = 0
        total_train = 0
        correct_test = 0
        total_test = 0

        
        # Training phase
        net.train()
        for batch, train_data in enumerate(tqdm(x_train, desc=f"Epoch {e+1}/{epochs}", leave=False)):
            # Get the inputs
            ip, op = train_data, y_train[batch]

            # Ensure target has the same shape as input
            if op.dim() == 3:  # Check if target lacks batch dimension
                op = op.unsqueeze(0)
                
            optimizer.zero_grad()

            model_op = net(ip)
            loss = criterion(model_op, op)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            # Calculate training accuracy
            _, predicted = torch.max(model_op.data, 1)
            _, labels = torch.max(op.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # Testing phase
        net.eval()
        with torch.no_grad():
            for batch_test, test_data in enumerate(tqdm(x_test, desc="Testing", leave=False)):
                ip_test, op_test = test_data, y_test[batch_test]
                model_op = net(ip_test)

                # Ensure the output shape matches the target shape
                if model_op.size() != op_test.size():
                    model_op = model_op.view_as(op_test)

                loss_test = criterion(model_op, op_test)
                test_loss += loss_test.item()

                # Calculate testing accuracy
                _, predicted_test = torch.max(model_op.data, 1)
                _, labels_test = torch.max(op_test.data, 1)
                total_test += labels_test.size(0)
                correct_test += (predicted_test == labels_test).sum().item()

        train_loss_container.append(train_loss)
        test_loss_container.append(test_loss)
        train_acc = 100 * correct_train / total_train
        test_acc = 100 * correct_test / total_test
        train_acc_container.append(train_acc)
        test_acc_container.append(test_acc)

        # Print epoch summary
        print(f'EPOCH: {e+1} | Train_loss: {train_loss:.4f} | Train_acc: {train_acc:.2f}% | 'f'Test_loss: {test_loss:.4f} | Test_acc: {test_acc:.2f}%')

    print('\nFinished Training')

    return net, train_loss_container, test_loss_container, train_acc_container, test_acc_container

def plot_losses(train_loss_container, test_loss_container):
    # Plotting the train and test loss as a function of epochs
    f, ax = plt.subplots(2, 1)
    ax[0].set_title('Training Loss')
    ax[0].plot(train_loss_container)
    ax[1].set_title('Test Loss')
    ax[1].plot(test_loss_container)
    f.tight_layout()
    plt.show()

def plot_accuracies(train_acc_container, test_acc_container):
    # Plotting the train and test accuracy as a function of epochs
    f, ax = plt.subplots(2, 1)
    ax[0].set_title('Training Accuracy')
    ax[0].plot(train_acc_container)
    ax[1].set_title('Test Accuracy')
    ax[1].plot(test_acc_container)
    f.tight_layout()
    plt.show()

