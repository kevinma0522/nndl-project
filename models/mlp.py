import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPClassifier(nn.Module):
    def __init__(self, num_super_classes, num_sub_classes):
        super(MLPClassifier, self).__init__()
        self.input_size = 64 * 64 * 3
        self.fc1 = nn.Linear(self.input_size, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(4096, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(2048, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.dropout3 = nn.Dropout(0.3)
        self.fc_super = nn.Linear(1024, num_super_classes)
        self.fc_sub = nn.Linear(1024, num_sub_classes)
    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(F.relu(self.bn3(self.fc3(x))))
        super_class_output = torch.sigmoid(self.fc_super(x))
        sub_class_output = torch.sigmoid(self.fc_sub(x))
        return super_class_output, sub_class_output 