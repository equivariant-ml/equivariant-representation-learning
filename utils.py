import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
