import os
import random

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision import datasets


def spliting_data(folder_name):
    list_image_name = []
    for image_name in os.listdir(folder_name):
        # print(image_name)
        list_image_name.append(image_name)
    random.shuffle(list_image_name)
    # print(list_image_name)
    train_set = []
    test_set = []
    for i in range(0, 1300):
        train_set.append(list_image_name[i])
    for j in range(1300, len(list_image_name)):
        test_set.append(list_image_name[j])
    for name in train_set:
        image = cv2.imread(folder_name + '/' + name)
        cv2.imwrite('classifier_data/train/' + folder_name.split('/')[1] + '/' + name + '.png', image)
    for name in test_set:
        image = cv2.imread(folder_name + '/' + name)
        cv2.imwrite('classifier_data/test/' + folder_name.split('/')[1] + '/' + name + '.png', image)


# spliting_data('classifier_data/with_bolt')
# spliting_data('classifier_data/without_bolt')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_datasets(train_path, test_path):

    data_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_path, transform=data_transforms)
    # print(len(train_data))
    test_data = datasets.ImageFolder(test_path, transform=data_transforms)
    # print(len(test_data))
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=True, num_workers=2)

    print(trainloader.dataset.classes)
    return trainloader, testloader


def train(trainloader):
    model = models.resnet34(pretrained=True)
    num_ftrs = model.fc.in_features
    # print(num_ftrs)
    model.fc = nn.Sequential(
                            nn.Dropout(0.5),
                            nn.Linear(num_ftrs, 2)
                            )
    ct = 0
    for child in model.children():
        ct += 1
        if ct < 3:
            for param in child.parameters():
                param.requires_grad = False
    print(model)
    model = nn.DataParallel(model)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(20):  # loop over the dataset multiple times
        print('training for epoch:' + str(epoch))
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # print(running_loss)
            if i % 200 == 199:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

    print('Finished Training')
    PATH = './saved_model/resnet34.pth'
    torch.save(model.state_dict(), PATH)


def test(testloader, PATH):

    model = models.resnet34()
    num_ftrs = model.fc.in_features
    # print(num_ftrs)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 2)
    )
    model = nn.DataParallel(model)
    model.to(device)
    model.load_state_dict(torch.load(PATH))
    correct = 0
    total = 0
    # print(model)
    print('testing the model')
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))


if __name__ == "__main__":
    trainloader, testloader = load_datasets('./classifier_data/train', './classifier_data/test')
    print('data loaded')
    train(trainloader)
    test(testloader, './saved_model/resnet34.pth')
