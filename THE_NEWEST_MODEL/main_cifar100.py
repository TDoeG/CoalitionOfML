import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from model import Net

def main(Batch_Size, Epochs, Learning_Rate, Experiment):
    # Define transformations for the training and test sets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    # Data augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),  # CIFAR-100 mean and std
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),  # CIFAR-100 mean and std
    ])


    # Load the CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR100(root='./THE_NEWEST_MODEL/data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, Batch_Size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./THE_NEWEST_MODEL/data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, Batch_Size, shuffle=False, num_workers=2)

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

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), Learning_Rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,patience=5,verbose=True)

    for epoch in range(Epochs):
        running_loss=0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/100:.3f}')
                running_loss = 0.0
                scheduler.step(running_loss)
    print('Finished Training')

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f}%')

    torch.save(net.state_dict(), f'./THE_NEWEST_MODEL/Saved_Models/{accuracy:.2f}_{Experiment}cifar10_cnn.pth')

    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # Print images
    plt.imshow(torchvision.utils.make_grid(images).permute(1, 2, 0) * 0.5 + 0.5)
    plt.show()

    # Show ground truth and predicted labels
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]}' for j in range(4)))
    outputs = net(images)

    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join(f'{classes[predicted[j]]}' for j in range(4)))

if __name__ == '__main__':
    main(
        Batch_Size=100,
        Epochs=50,
        Learning_Rate=0.001,
        Experiment=4
    )