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

    for e in tqdm(range(epochs), desc="Training Progress", unit="epoch"):
        train_loss = 0.0
        test_loss = 0.0
        
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

        # Testing phase
        net.eval()
        with torch.no_grad():
            for batch_test, test_data in enumerate(tqdm(x_test, desc="Testing", leave=False)):
                ip_test, op_test = test_data, y_test[batch_test]
                model_op = net(ip_test)
                loss_test = criterion(model_op, op_test)
                test_loss += loss_test.item()

        train_loss_container.append(train_loss)
        test_loss_container.append(test_loss)
        print(f'\rEPOCH: {e+1} | Train_loss: {train_loss:.4f} | Test_loss: {test_loss:.4f}', end='')

    print('\nFinished Training')

    return net, train_loss_container, test_loss_container

def plot_losses(train_loss_container, test_loss_container):
    # Plotting the train and test loss as a function of epochs
    f, ax = plt.subplots(2, 1)
    ax[0].set_title('Training Loss')
    ax[0].plot(train_loss_container)
    ax[1].set_title('Test Loss')
    ax[1].plot(test_loss_container)
    f.tight_layout()
    plt.show()
