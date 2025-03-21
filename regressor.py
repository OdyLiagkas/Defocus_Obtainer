import torch
import torch.nn as nn
import torch.nn.functional as F

class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        # Convolutional layers
        # ORIGINAL: 1x128x128
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)  #4x64x64
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)  #8x32x32
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  #16x16x16
        self.conv4 = nn.Conv2d(16, 32, kernel_size=3, padding=1)   #32x8x8
        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, padding=1)     #64x4x4

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Halves spatial dimensions

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(64*4*4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1) # Output a single value

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 + ReLU + Pooling -> Output: 4x100x100
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 + ReLU + Pooling -> Output: 8x50x50
        x = self.pool(F.relu(self.conv3(x)))  # Conv3 + ReLU + Pooling -> Output: 16x25x25
        x = self.pool(F.relu(self.conv4(x)))  # Conv4 + ReLU + Pooling -> Output: 32x12x12
        x= self.pool(F.relu(self.conv5(x)))   # Conv5 + ReLU + Pooling -> Output: 64x6x6
        x = self.flatten(x)  # Flatten to 1D -> Output: 64 * 6 * 6 = 2,304
        x = F.relu(self.fc1(x))  # Fully connected layer 1
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x)) # Sigmoid activation for output between 0 and 1
        return x

