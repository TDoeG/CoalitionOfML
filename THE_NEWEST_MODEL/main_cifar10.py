import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import Net

def main(Batch_Size, Epochs, Learning_Rate, Experiment):
    # Define transformations for the training and test sets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the CIFAR-10 dataset
    dataset = torchvision.datasets.CIFAR10(root='./THE_NEWEST_MODEL/data', train=True, download=True, transform=transform)
        # trainloader = torch.utils.data.DataLoader(trainset, Batch_Size, shuffle=True, num_workers=2)
        # testset = torchvision.datasets.CIFAR10(root='./THE_NEWEST_MODEL/data', train=False, download=True, transform=transform)
        # testloader = torch.utils.data.DataLoader(testset, Batch_Size, shuffle=False, num_workers=2)

    # Splits dataset into 70% train, 15% , 15% validation
    train_len = int(len(dataset) * 0.8)
    test_len = int(len(dataset) * 0.1)
    val_len = len(dataset) - train_len - test_len
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, test_len, val_len])
    trainloader = torch.utils.data.DataLoader(train_dataset, Batch_Size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_dataset, Batch_Size, shuffle=False, num_workers=2)
    valLoader = torch.utils.data.DataLoader(val_dataset, Batch_Size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), Learning_Rate)

    correct = 0
    total = 0

    for epoch in range(Epochs):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Initialize tqdm progress bar
        progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{Epochs}", unit="batch")
        
        for i, data in enumerate(progress_bar):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update tqdm with loss and accuracy information
            progress_bar.set_postfix(loss=running_loss/(i+1), accuracy=100.*correct/total)

        # Validation accuracy after each epoch
        net.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data in valLoader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_accuracy = 100. * val_correct / val_total
        print(f"Validation Accuracy after Epoch {epoch+1}: {val_accuracy:.2f}%")
    print('Finished Training')

    correct = 0
    total = 0

    net.eval()
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
        Batch_Size=64,
        Epochs=20,
        Learning_Rate=0.001,
        Experiment=6
    )

    # Experiment 3 with 10 epochs, weighted decay 1e-4 | Acc: 64.04%
    # Experiment 3 with 20 epochs | Acc: 63.19%
    # Experiment 3 with 20 epochs, no scheduler | Acc: 71.73%
    # Experiment 3 with 20 epochs, LR 0.0001 | Acc: 63.92%
    # Experiment 4 with 20 epochs | Acc: 71.61%
    # Experiment 4 with 20 epochs | Acc: 70.05%
    # Experiment 5 with same settings, diff model | Acc: 77.83%!! USING THIS MODEL
    # Experiment 5 with weighted decay & validation set | Acc: 76.34%
    # Experiment 5 w/o weighted decay & validation set | Acc: 77.98%
    # Experiment 6 with 64 batch size 70, 15, 15 split | Acc: 77.59%
    # Experiment 6 with 64 batch size, 80, 10, 10 split | Acc: 

