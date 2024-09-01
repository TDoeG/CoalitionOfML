import matplotlib.pyplot as plt
import numpy as np

def plot_losses(train_loss_container, test_loss_container):
    # Create a figure and a set of subplots
    f, ax = plt.subplots(2, 1, figsize=(8, 6))  # 2 rows, 1 column, with a specified figure size

    # Plot the training loss
    # ax[0] is the first subplot, which will display the training loss over epochs
    ax[0].set_title('Training Loss')  # Set the title of the first subplot
    ax[0].plot(train_loss_container, color='blue', linestyle='-', marker='o', markersize=4)  # Plot the training loss with a line plot
    ax[0].set_xlabel('Epochs')  # Label the x-axis as 'Epochs'
    ax[0].set_ylabel('Loss')  # Label the y-axis as 'Loss'
    ax[0].grid(True)  # Enable the grid for better visualization

    # Plot the testing loss
    # ax[1] is the second subplot, which will display the test loss over epochs
    ax[1].set_title('Test Loss')  # Set the title of the second subplot
    ax[1].plot(test_loss_container, color='red', linestyle='-', marker='o', markersize=4)  # Plot the test loss with a line plot
    ax[1].set_xlabel('Epochs')  # Label the x-axis as 'Epochs'
    ax[1].set_ylabel('Loss')  # Label the y-axis as 'Loss'
    ax[1].grid(True)  # Enable the grid for better visualization

    # Adjust the layout so that subplots fit into the figure area without overlapping
    f.tight_layout()

    # Display the plots
    plt.show()

def visualize_inputs(x_test, mean, std):
    # Create a figure with a 3x3 grid of subplots
    f, ax = plt.subplots(3, 3, figsize=(8, 8))  # 3 rows, 3 columns, with a specified figure size

    # Set the overall title for the figure
    f.suptitle('Input', fontsize=16)  # Larger font size for the main title

    # Loop over the rows (i) and columns (j) of the subplot grid
    for i in range(3):
        for j in range(3):
            # Get the corresponding test image from x_test and unnormalize it
            tmp = mean.item() + x_test[i][j].detach().cpu().squeeze().numpy() * std.item()
            
            # Reshape the image to 32x32 and convert it to uint8 for display
            img = tmp.reshape(32, 32).astype(np.uint8)
            
            # Display the image in grayscale using imshow in the (i, j) position of the grid
            ax[i, j].imshow(img, cmap='gray')
            
            # Optionally, turn off axis labels and ticks for a cleaner look
            ax[i, j].axis('off')

    # Adjust layout to ensure that subplots do not overlap
    f.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout while keeping space for the title

    # Display the figure with the 3x3 grid of images
    plt.show()

def visualize_predictions(model, x_test):
    # Create a figure with a 3x3 grid of subplots
    f, ax = plt.subplots(3, 3, figsize=(8, 8))  # 3 rows, 3 columns, with a specified figure size

    # Set the overall title for the figure
    f.suptitle('Predictions', fontsize=16)  # Larger font size for the main title

    # Loop over the rows (i) and columns (j) of the subplot grid
    for i in range(3):
        for j in range(3):
            # Get the model prediction for the i-th image in the test set
            tmp = model(x_test[i]).detach().cpu().numpy()[j].reshape(32, 32, 3) * 255
            
            # Convert the pixel values to uint8 format for display
            img = tmp.astype(np.uint8)
            
            # Display the predicted image using imshow in the (i, j) position of the grid
            ax[i, j].imshow(img)
            
            # Optionally, turn off axis labels and ticks for a cleaner look
            ax[i, j].axis('off')

    # Adjust layout to ensure that subplots do not overlap
    f.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout while keeping space for the title

    # Display the figure with the 3x3 grid of images
    plt.show()

def visualize_ground_truth(y_test):
    # Create a figure and a 3x3 grid of subplots
    f, ax = plt.subplots(3, 3, figsize=(10, 10))  # Added figsize for better clarity
    f.suptitle('Ground-Truth', fontsize=16)  # Added fontsize for better visibility

    # Loop through each subplot in the 3x3 grid
    for i in range(3):
        for j in range(3):
            # Convert the tensor to a numpy array and scale it
            tmp = y_test[i][j].detach().cpu().numpy() * 255
            
            # Reshape the array to match the image dimensions (32x32x3)
            img = tmp.reshape(32, 32, 3).astype(np.uint8)
            
            # Display the image in the corresponding subplot
            ax[i, j].imshow(img)
            
            # Remove axis ticks for a cleaner look
            ax[i, j].axis('off')

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the space to fit the suptitle
    plt.show()