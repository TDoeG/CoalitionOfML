# Standard/Neceessary Libraries
import os
import torch
import matplotlib.pyplot as plt

# Local application/library imports
from data_processing import (
    load_cifar10_dataset,
    create_batches,
    split_data,
    normalize_data,
    transform_and_create_torch_tensors
)
from training import train_model
from visualization import (
    plot_losses,
    visualize_inputs,
    visualize_predictions,
    visualize_ground_truth
)
from model import ConvNet



# Set parameters-----------------------------------------------------------------------------------------------------------------------
root_dir = './assets/cifar-10/'
batch_size = 100
learning_rate = 0.0001
epochs = 1
Exp = 1

# Load and process data---------------------------------------------------------------------------------------------------------------
cifar_dataset = load_cifar10_dataset(root_dir) # Loads dataset
ip_data, op_data = create_batches(cifar_dataset, batch_size) # Creates batches
x_train, y_train, x_test, y_test = split_data(ip_data, op_data) # Creates train and test from batches

# Random Testing #1-------------------------------------------------------------------------------------------------------------------
f, ax_1 = plt.subplots(1,2)
ax_1[0].imshow(x_train[0][0], cmap='gray')
ax_1[1].imshow(y_train[0][0])
ax_1[0].set_title('x_train'), ax_1[1].set_title('y_train')

# Load and process data 2-------------------------------------------------------------------------------------------------------------
x_train, y_train, x_test, y_test = map(lambda data: transform_and_create_torch_tensors(data, batch_size), [x_train, y_train, x_test, y_test])
# x_train, y_train, x_test, y_test = map(transform_and_create_torch_tensors, [x_train, y_train, x_test, y_test], batch_size) # Transforms data
x_train, x_test, y_train, y_test, mean, std = normalize_data(x_train, x_test, y_train, y_test)

# Train the model and plot the losses-------------------------------------------------------------------------------------------------
model, train_loss_container, test_loss_container = train_model(x_train, y_train, x_test, y_test, batch_size, learning_rate, epochs)

# Save the trained model--------------------------------------------------------------------------------------------------------------
# Ensure the directory for saving the model exists
save_dir = './saved_model/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Construct the path for saving the model file and save model at path
model_path = os.path.join(save_dir, f'G2C_Exp{Exp}Epoch{epochs}.pth')
torch.save(model.state_dict(), model_path)
print(f'Model saved at {model_path}')

# Plot the losses---------------------------------------------------------------------------------------------------------------------
plot_losses(train_loss_container, test_loss_container)

# Loads the saved model---------------------------------------------------------------------------------------------------------------
#model_name = f'G2C_Exp{Exp}Epoch{epochs}.pth'
model_name = 'model(1).pth'
model_path = f'./saved_model/{model_name}'
model = ConvNet(batch_size)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
model.eval()

# Plot the input images, predictions, and ground truth--------------------------------------------------------------------------------
visualize_inputs(x_test, mean, std)
visualize_predictions(model, x_test)
visualize_ground_truth(y_test)