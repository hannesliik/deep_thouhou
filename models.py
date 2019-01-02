import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(torch.nn.Module):

    def __init__(self, frames_memory, num_actions, size_x, size_y):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(frames_memory * 3, 20, 3, dilation=8)
        self.conv2 = nn.Conv2d(20, 20, 3, dilation=4)
        self.conv3 = nn.Conv2d(20, 20, 3, dilation=2)
        self.conv4 = nn.Conv2d(20, 20, 3, dilation=1)
        #self.fc1 = nn.Linear(20 * (size_x - 8) * (size_y - 8), num_actions)
        self.fc1 = nn.Linear(306000, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x.float()))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc1(x)

