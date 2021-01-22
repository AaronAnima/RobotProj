import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
from torch.distributions import Normal


def conv2d_size_out(size, kernel_size=4, stride=2, padding=1):
    return (size - kernel_size + 2 * padding) // stride + 1


class Classifier(nn.Module):
    def __init__(self, max_action, h, w, action_dim, order, neu_dim, ch, depth, img_c):
        super(Classifier, self).__init__()
        # ch = 16
        neu_size = neu_dim
        self.max_action = max_action
        self.order = order
        self.img_c = img_c
        convw = w
        convh = h
        # get convh & convw after convs
        for _ in range(depth):
            convw = conv2d_size_out(convw, 4, 2, 1)
            convh = conv2d_size_out(convh, 4, 2, 1)
        linear_input_size = convw * convh * ch * (2 ** (depth - 1))
        cur_dim = self.img_c
        next_dim = ch
        backbone_layers = []
        for idx in range(depth):
            backbone_layers.append(nn.Conv2d(cur_dim, next_dim, kernel_size=4, stride=2, padding=1))
            backbone_layers.append(nn.BatchNorm2d(ch * (2 ** idx)))
            backbone_layers.append(nn.ReLU())
            cur_dim = next_dim
            next_dim *= 2
        self.backbone = nn.Sequential(*backbone_layers)
        self.head_1 = nn.Linear(linear_input_size * order, neu_size)
        self.head_2 = nn.Linear(neu_size, action_dim)
    
    def forward(self, x):
        obs = [x[:, i * self.img_c: (i + 1) * self.img_c, :, :] for i in range(self.order)]
        feature = torch.cat([self.backbone(input_obs).view(input_obs.size(0), -1) for input_obs in obs], dim=1)
        feature = F.relu(self.head_1(feature))
        feature = self.head_2(feature)
        return self.max_action * torch.tanh(feature)
