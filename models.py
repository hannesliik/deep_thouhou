import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(torch.nn.Module):

    def __init__(self, frames_memory, num_actions, size_x, size_y):
        super(Model, self).__init__()
        self.conv1_large = nn.Conv2d(frames_memory * 3, 20, 3, dilation=8, padding=8)
        self.conv1_med = nn.Conv2d(frames_memory * 3, 20, 3, dilation=4, padding=4)
        self.conv1_small = nn.Conv2d(frames_memory * 3, 20, 3, dilation=2, padding=2)
        self.conv1_tiny = nn.Conv2d(frames_memory * 3, 20, 3, dilation=1, padding=1)

        self.conv2 = nn.Conv2d(80, 20, 3, dilation=4)
        self.conv3 = nn.Conv2d(20, 20, 3, dilation=2)
        self.conv4 = nn.Conv2d(20, 20, 3, dilation=1)
        #self.fc1 = nn.Linear(20 * (size_x - 8) * (size_y - 8), num_actions)
        self.fc1 = nn.Linear(394320, num_actions)

    def forward(self, x):
        x = x.float()
        layer1 = F.relu(torch.cat((self.conv1_large(x), self.conv1_med(x), self.conv1_small(x), self.conv1_tiny(x)), dim=1))
        x = F.relu(self.conv2(layer1))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc1(x)

