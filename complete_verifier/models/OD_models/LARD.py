import os

import torch.nn.functional as F
from torch import nn
import torch
import IoU as IoU

torch_model_seq = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=(3,3), stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=(3,3), stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 128, kernel_size=(3,3), stride=2, padding=1),
    nn.ReLU(),
    nn.Flatten(), #131072
    nn.Linear(131072, 128),
    nn.ReLU(),
    nn.Linear(128,128),
    nn.ReLU(),
    nn.Linear(128,4),
)


class CustomModelLARD(nn.Module):
    def __init__(self, original_model):
        super(CustomModelLARD, self).__init__()
        self.linear = nn.Linear(1, 3*256*256)
        self.model = original_model

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 3, 256, 256)  
        x = self.model(x)
        return x



class Neural_network_LARD(nn.Module):
    def __init__(self):
        super(Neural_network_LARD, self).__init__()
        self.conv0 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=2, padding=1)
        self.conv2 =nn.Conv2d(32, 64, kernel_size=(3,3), stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=2, padding=1)
        self.flatten = nn.Flatten() #131072
        self.linear7 = nn.Linear(131072, 128)
        self.linear9 = nn.Linear(128,128)
        self.linear11 = nn.Linear(128,4)
        self.relu = nn.ReLU()
   
    def forward(self, x):
        x = self.conv0(x)
        x =  self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.linear7(x)
        x = self.relu(x)
        x = self.linear9(x)
        x = self.relu(x)
        x = self.linear11(x)
        return(x)


class Neural_network_LARD_BrightnessContrast(nn.Module):
    def __init__(self):
        super(Neural_network_LARD_BrightnessContrast, self).__init__()
        self.linear_perturbation = nn.Linear(1,3*256*256) # to check
        self.conv0 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=2, padding=1)
        self.conv2 =nn.Conv2d(32, 64, kernel_size=(3,3), stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=2, padding=1)
        self.flatten = nn.Flatten() #131072
        self.linear7 = nn.Linear(131072, 128)
        self.linear9 = nn.Linear(128,128)
        self.linear11 = nn.Linear(128,4)
        self.relu = nn.ReLU()
   
    def forward(self, x):
        x = self.linear_perturbation(x)
        x = x.view(-1, 3, 256, 256)  
        x = self.conv0(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.linear7(x)
        x = self.relu(x)
        x = self.linear9(x)
        x = self.relu(x)
        x = self.linear11(x)
        return(x)

# build new model, which call to Neural_network_LARD_IoU, and add IoU block to calculate IoU
# The model should get also the gt coordinates as input
class Neural_network_LARD_IoU(nn.Module):
    def __init__(self, tau_min=0.5, tau_max=0.5):
        super(Neural_network_LARD_IoU, self).__init__()
        self.model = Neural_network_LARD()
        self.iou = IoU.IoU(tau_min, tau_max)

    def forward(self, x, gt=None):
        # x is the input image
        # gt is the ground truth coordinates - bounding box
        # _logits = self.model(x.float())
        #
        # # Calculate the required modifications without in-place operations
        # col2 = _logits[:, 2] - _logits[:, 0]
        # col3 = _logits[:, 3] - _logits[:, 1]
        #
        # # Prepare swapped columns
        # col0 = _logits[:, 1]
        # col1 = _logits[:, 0]
        #
        # # Concatenate all columns
        # logits = torch.cat((col0.unsqueeze(1), col1.unsqueeze(1), col2.unsqueeze(1), col3.unsqueeze(1)), dim=1)

        logits = self.model(x.float())
        # convert logits from yxyx to yxhw
        # logits[:, 2], logits[:, 3] = logits[:, 2] - logits[:, 0], logits[:, 3] - logits[:, 1]
        # logits[:, [0, 1]] = logits[:, [1, 0]]

        # if gt is None, gt = logits
        if gt is None:
            gt = logits

        z = self.iou(torch.cat((logits, gt), dim=1))
        return z

        import matplotlib.pyplot as plt
        plt.imshow(x[0].permute(1, 2, 0).cpu().detach().numpy())
        _gt = gt.cpu().detach().numpy()
        # gt form as [[x1, y1, x2, y2]]
        plt.plot([_gt[0][0], _gt[0][2], _gt[0][2], _gt[0][0], _gt[0][0]],
                 [_gt[0][1], _gt[0][1], _gt[0][3], _gt[0][3], _gt[0][1]], 'g')
        # draw logits in red
        _logits = logits.cpu().detach().numpy()
        plt.plot(
            [_logits[0][0], _logits[0][2], _logits[0][2], _logits[0][0], _logits[0][0]],
            [_logits[0][1], _logits[0][1], _logits[0][3], _logits[0][3], _logits[0][1]], 'r')