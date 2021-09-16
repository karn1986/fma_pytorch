import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        ##https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#define-a-convolutional-neural-network
        # input - 128 x 128
        self.conv1 = nn.Conv2d(1, 8, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(8)
        #pool - 64 x 64
        self.conv2 = nn.Conv2d(8, 16, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(16)
        #pool - 32 x 32
        self.conv3 = nn.Conv2d(16, 32, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(32)
        #pool - 16 x 16
        self.conv4 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(64)
        #pool - 8 x 8
        self.conv5 = nn.Conv2d(64,64,2,2)
        self.bn5 = nn.BatchNorm2d(64)
        # 4 x 4
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64*4*4, 192)
        self.bn6 = nn.BatchNorm1d(192)
        self.fc2 = nn.Linear(192,16)


    def forward(self, x):
        outs = self.conv1(x)
        outs = nn.functional.relu(self.bn1(outs))
        outs = self.pool(outs)
        #
        outs = self.conv2(outs)
        outs = nn.functional.relu(self.bn2(outs))
        outs = self.pool(outs)
        #
        outs = self.conv3(outs)
        outs = nn.functional.relu(self.bn3(outs))
        outs = self.pool(outs)
        #
        outs = self.conv4(outs)
        outs = nn.functional.relu(self.bn4(outs))
        outs = self.pool(outs)
        #
        outs = self.conv5(outs)
        outs = nn.functional.relu(self.bn5(outs))
        #
        B = outs.shape[0]
        outs = outs.reshape((B,-1))
        outs = self.fc1(outs)
        outs = nn.functional.relu(self.bn6(outs))
        #
        outs = self.fc2(outs)
        return outs