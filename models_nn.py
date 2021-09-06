import torch
from torch import nn, load
from torch.nn import functional as F
from torchvision.utils import save_image
import torch.nn.functional as F
import torchvision.models as torch_models
from torch.autograd import Variable
import torchvision.transforms as transforms
from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply, euler_angles_to_matrix

import numpy as np
from functools import reduce
import nn_utils

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class AE(nn.Module):
    def __init__(self, encoder, decoder, latent_dim, action_dim, encoding='tanh', device='cuda'):
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.action_dim = action_dim

        self.encoding = encoding
        self.device = device


    def encode(self, x):
        encoded = self.encoder(x)
        return encoded

    def decode(self, z):
        return self.decoder(z)

    def encode_pose(self, x):
        z = self.encode(x)
        return self.pose_activation(z[:, :self.action_dim])

    def pose_activation(self, z):
        if self.encoding == 'tanh':
            return torch.tanh(z)
        elif self.encoding == 'normalize':
            return nn_utils.normalize(z)
        elif self.encoding == 'non':
            return z
        elif self.encoding == 'euler':
            t = F.tanh(z)
            t = t * torch.Tensor([[np.pi, np.pi/2, np.pi]]).to(self.device)
            return t
        elif self.encoding == 'angle':
            return nn_utils.normalize(z)
        elif self.encoding == 'trans_angle':
            rot_part = nn_utils.normalize(z[:,2:])
            return  torch.cat((z[:, :2], rot_part), -1)


    def forward(self, x):
        z = self.encode(x)

        pose = self.pose_activation(z[:, :self.action_dim])
        extra = z[:, self.action_dim:]

        total = torch.cat((pose, extra), -1)
        return self.decode(total), z



class AE_CNN(AE):                           #Works on 64x64 images
    def __init__(self, latent_dim, nc, action_dim, encoding):

        encoder = nn.Sequential(
            nn.Conv2d(nc, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 1),
            nn.ReLU(True),
            View([-1, 256]),
            nn.Linear(256, latent_dim),
        )
        decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            View((-1, 256, 1, 1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, nc, 4, 2, 1),
            nn.Sigmoid()
            )
        super(AE_CNN, self).__init__(encoder, decoder, latent_dim, action_dim, encoding)




if __name__ == "__main__":
    pass
