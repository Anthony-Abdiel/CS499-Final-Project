import torch.nn as nn


class LightweightCNN(nn.Module):
    def __init__(self):
        super().__init__()

        #defining the convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        #defining reused Relu, pooling, and flattening objects
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        #Defining the fully connected layers
        self.fc1 = nn.Linear(3136, 128)
        self.fc2 = nn.Linear(128, 8)

    
    def forward(self, data):
        data = self.conv1(data)
        data = self.relu(data)
        data = self.pool(data)

        data = self.conv2(data)
        data = self.relu(data)
        data = self.pool(data)

        data = self.conv3(data)
        data = self.relu(data)

        data = self.flatten(data)

        data = self.fc1(data)
        data = self.relu(data)

        data = self.fc2(data)

        return data