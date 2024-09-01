import torch
import os
from data_processing import load_cifar10_dataset, create_batches, split_data, normalize_data
from training import train_model, plot_losses, plot_accuracies, visualize_inputs, visualize_predictions, visualize_ground_truth


# Set parameters-----------------------------------------------------------------------------------------------------------------------
root_dir = './New_Model/assets/cifar-10/'
batch_size = 100
learning_rate = 0.0001
epochs = 1
Exp = 1

# Load and process data---------------------------------------------------------------------------------------------------------------
cifar_dataset, num_samples = load_cifar10_dataset(root_dir, batch_size)
ip_data, op_data = create_batches(cifar_dataset, num_samples, batch_size)
x_train, y_train, x_test, y_test = split_data(ip_data, op_data)
x_train, x_test, y_train, y_test = normalize_data(x_train, x_test, y_train, y_test)

# Train the model and plot the losses-------------------------------------------------------------------------------------------------
model, train_loss_container, test_loss_container, train_acc_container, test_acc_container = train_model(x_train, y_train, x_test, y_test, batch_size, learning_rate, epochs)

# Save the trained model--------------------------------------------------------------------------------------------------------------
model_save_path = 'trained_model.pth'

# Makes the folder if it doesn't exist
if not os.path.exists('./saved_model/'):
    os.makedirs('./saved_model/')
        
# Adds model to the model path--------------------------------------------------------------------------------------------------------
model_path = os.path.join('./saved_model', f'G2C_Exp{Exp}Epoch{epochs}.pth')
torch.save(model.state_dict(), model_path)
print(f'Model saved at {model_path}')

# Plot the losses---------------------------------------------------------------------------------------------------------------------
plot_losses(train_loss_container, test_loss_container)
plot_accuracies(train_acc_container, test_acc_container)

# Plot the input images, predictions, and ground truth--------------------------------------------------------------------------------
visualize_inputs(x_test)
visualize_predictions(model, x_test)
visualize_ground_truth(y_test)