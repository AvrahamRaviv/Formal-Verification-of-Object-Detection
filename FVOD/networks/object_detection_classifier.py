import torch
import torch.nn.functional as F
from torch import nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn.functional as F
from torch import nn as nn

class NeuralNetwork_OL_v2_old(nn.Module):
    '''
    New convolutional model (v2)
    '''

    def __init__(self):
        super(NeuralNetwork_OL_v2_old, self).__init__()
        seed = 0
        torch.manual_seed(seed)
        padding = 1

        self.conv0 = nn.Conv2d(1, 16, 3, padding=padding)  # 3x3 filters w/ same padding
        self.pool0 = nn.MaxPool2d(2, stride=2)
        # output shape : 15x15x16
        self.conv1 = nn.Conv2d(16, 16, 3, padding=padding)  # 3x3 filters w/ same padding

        self.pool1 = nn.MaxPool2d(2, stride=2)
        # output shape : 8x8x16
        self.flatten = nn.Flatten()
        # output shape : 1024
        # HERE CHECK RIGHT SIZE FROM FLATTEN TO LINEAR

        self.linear_relu_stack = nn.Linear(7744, 256)
        self.linear_relu_stack = nn.Linear(7744, 256)
        self.linear = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv0(x)
        x = F.relu(self.pool0(x))
        x = self.conv1(x)
        x = F.relu(self.pool1(x))
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        x = F.relu(x)
        logits = self.linear(x)

        return logits
