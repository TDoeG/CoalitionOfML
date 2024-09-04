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

    # Load the CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./THE_NEWEST_MODEL/data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, Batch_Size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./THE_NEWEST_MODEL/data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, Batch_Size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), Learning_Rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

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
                scheduler.step()
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
        Epochs=20,
        Learning_Rate=0.001,
        Experiment=3
    )

    # Experiement 3 with 10 epochs, weighted decay 1e-4 | Acc: 64.04%
    # Experiement 3 with 20 epochs | Acc: 