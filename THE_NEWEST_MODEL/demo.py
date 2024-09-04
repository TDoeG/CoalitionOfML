import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
from model import Net  # Assuming model.py is in the same directory

# Load the CIFAR-10 or CIFAR-100 dataset
def load_dataset(dataset_name):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),  # CIFAR-100 mean and std
    ])
    if dataset_name == 'CIFAR100':
        dataset = torchvision.datasets.CIFAR100(root='./THE_NEWEST_MODEL/data', train=False, download=True, transform=transform)
        classes = ('apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
           'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
           'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
           'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 
           'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 
           'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 
           'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 
           'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 
           'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 
           'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 
           'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 
           'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 
           'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 
           'woman', 'worm')
    else:
        dataset = torchvision.datasets.CIFAR10(root='./THE_NEWEST_MODEL/data', train=False, download=True, transform=transform)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return dataset, classes

# Load the saved model
def load_model(model_path):
    model = Net()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

# Grab 3 random images from the dataset and get their predicted classes
def predict_random_images(model, dataset, classes, num_images=3):
    images, labels, predictions = [], [], []
    for _ in range(num_images):
        random_idx = random.randint(0, len(dataset) - 1)
        image, label = dataset[random_idx]
        image_tensor = image.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)

        images.append(image)
        labels.append(label)
        predictions.append(predicted.item())
    
    return images, labels, predictions

# Plot the images with their actual and predicted classes
def plot_results(images, actual_classes, predicted_classes, classes):
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    for i in range(len(images)):
        image = images[i].permute(1, 2, 0) * 0.5 + 0.5  # Unnormalize
        axes[i].imshow(image)
        axes[i].set_title(f'Actual: {classes[actual_classes[i]]}\nPredicted: {classes[predicted_classes[i]]}')
        axes[i].axis('off')
    plt.show()

def main():
    # Load dataset and model
    dataset_name = 'CIFAR10'  # Change to 'CIFAR100' if you want to use CIFAR-100
    model_path = './THE_NEWEST_MODEL/Saved_Models/72.34_3cifar10_cnn.pth'  # Replace with your actual model path
    dataset, classes = load_dataset(dataset_name)
    model = load_model(model_path)

    while True:
        # Predict 3 random images and plot the results
        images, actual_classes, predicted_classes = predict_random_images(model, dataset, classes, num_images=3)
        plot_results(images, actual_classes, predicted_classes, classes)

        # Wait for user input
        user_input = input("Press 'N' to show 3 more images or any other key to exit: ").strip().lower()
        if user_input != 'n':
            break

if __name__ == '__main__':
    main()
