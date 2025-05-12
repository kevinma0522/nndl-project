import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(nn.Module):
    def __init__(self, num_super_classes, num_sub_classes):
        super(CNNClassifier, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(192)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(384)
        
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Calculate the size of flattened features
        self._to_linear = None
        self._get_conv_output_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 4096)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 2048)
        self.dropout2 = nn.Dropout(0.5)
        
        # Output layers for super-class and sub-class
        self.fc_super = nn.Linear(2048, num_super_classes)
        self.fc_sub = nn.Linear(2048, num_sub_classes)
        
    def _get_conv_output_size(self):
        # Create a dummy input to calculate the size
        x = torch.randn(1, 3, 64, 64)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))
        self._to_linear = x.shape[1] * x.shape[2] * x.shape[3]
        
    def forward(self, x):
        # Convolutional layers
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        # Output layers
        super_class_output = torch.sigmoid(self.fc_super(x))
        sub_class_output = torch.sigmoid(self.fc_sub(x))
        
        return super_class_output, sub_class_output 