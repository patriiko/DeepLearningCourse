import torch
import torch.nn as nn
import torch.nn.functional as F


class Net_Cifar(nn.Module):
    def __init__(self):
        super(Net_Cifar, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.maxpool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.maxpool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)

        self.maxpool3 = nn.MaxPool2d(2, 2)

        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.bn7 = nn.BatchNorm1d(512)

        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 512)
        self.bn8 = nn.BatchNorm1d(512)

        self.dropout3 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.maxpool1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.maxpool2(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.maxpool3(x)

        x = x.view(x.size(0), -1)

        x = self.dropout1(x)
        x = F.relu(self.bn7(self.fc1(x)))

        x = self.dropout2(x)
        x = F.relu(self.bn8(self.fc2(x)))

        x = self.dropout3(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

