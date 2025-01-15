import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Output: 32x200x200
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output: 64x100x100
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Output: 128x50x50
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Halves spatial dimensions
        
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 25 * 25, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 + ReLU + Pooling -> Output: 32x100x100
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 + ReLU + Pooling -> Output: 64x50x50
        x = self.pool(F.relu(self.conv3(x)))  # Conv3 + ReLU + Pooling -> Output: 128x25x25
        x = self.flatten(x)  # Flatten to 1D -> Output: 128 * 25 * 25
        x = F.relu(self.fc1(x))  # Fully connected layer 1
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

