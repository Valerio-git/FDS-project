import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # ----- Convolutional layers -----
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # ----- Fully connected -----
        # 128 -> 512 -> 256 -> num_classes
        self.fc1 = nn.Linear(128, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # ----- Conv block 1 -----
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ----- Conv block 2 -----
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ----- Conv block 3 -----
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Global Average Pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        # ----- Fully connected -----
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = self.relu(x)

        x = self.dropout(x)
        x = self.fc3(x)
        return x
