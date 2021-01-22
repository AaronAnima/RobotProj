import numpy as np
import torch
import random
import os
import gym
import math
import torch
from collections import namedtuple, deque
from torchvision import transforms as T
from PIL import Image
from torchvision.utils import save_image, make_grid


def exists_or_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    else:
        return True
    
    
def string2bool(item):
    return item == 'True'
