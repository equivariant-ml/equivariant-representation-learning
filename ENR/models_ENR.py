import torch
from torch import nn, load
from torch.nn import functional as F
from torchvision.utils import save_image
import torch.nn.functional as F
import torchvision.models as torch_models
from torch.autograd import Variable
import torchvision.transforms as transforms

import numpy as np
from functools import reduce
import utils
import utils.nn_utils as nn_utils

from ENR.models.submodels import ResNet2d, ResNet3d, Projection, InverseProjection
from ENR.models.rotation_layers import SphericalMask, Rotate3d
import ipdb

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class Print_shape(nn.Module):
    def __init__(self):
        super(Print_shape, self).__init__()

    def forward(self, tensor):
        print(tensor.shape)
        return tensor



class AE(nn.Module):
    def __init__(self, encoder, decoder):
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        encoded = self.encoder(x)
        return encoded

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, action):
        z = self.encode(x)

        #The matrix is processed by inverting, flipping rows and columns and adding a
        #column of zeros as the translational part
        matrix = torch.flip(action.transpose(-2, -1), dims=(-2, -1))
        translations = torch.zeros(matrix.shape[0], 3, 1, device=matrix.device)
        affine_matrix = torch.cat([matrix, translations], dim=2)

        #Because of some PyTorch mystery, align_corners should be False
        grid = F.affine_grid(affine_matrix, z.shape, align_corners=False)
        z_rotated = F.grid_sample(z, grid, align_corners=False)

        return self.decode(z_rotated)


class CNN(AE):                           #64x64
    def __init__(self, nc):

        encoder = nn.Sequential(
            nn.Conv2d(nc, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            View([-1, 4, 32, 32, 32]),
            nn.Conv3d(4, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv3d(32, 8, 4, 1, 1)
        )
        decoder = nn.Sequential(
            nn.ConvTranspose3d(8, 32, 4, 1, 1),
            nn.ReLU(True),
            nn.ConvTranspose3d(32, 4, 2, 2),
            nn.ReLU(True),
            View([-1, 128, 32, 32]),
            nn.ConvTranspose2d(128, 64, 4, 2, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 1, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, nc, 4, 1, 1),
            #Print_shape(),
            nn.Sigmoid()
            )
        super(CNN, self).__init__(encoder, decoder)


class CNN_naive(AE):                           #Works on 64x64 images
    def __init__(self, nc, grid_size=20):

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
            nn.Linear(256, grid_size**4),
            View([-1, grid_size, grid_size, grid_size, grid_size])
        )
        decoder = nn.Sequential(
            View([-1, grid_size**4]),
            nn.Linear(grid_size**4, 256),
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
        super(CNN_naive, self).__init__(encoder, decoder)



class NeuralRenderer(nn.Module):

    def __init__(self, img_shape, channels_2d, strides_2d, channels_3d,
                 strides_3d, num_channels_inv_projection, num_channels_projection,
                 mode='bilinear'):
        super(NeuralRenderer, self).__init__()
        self.img_shape = img_shape
        self.channels_2d = channels_2d
        self.strides_2d = strides_2d
        self.channels_3d = channels_3d
        self.strides_3d = strides_3d
        self.num_channels_projection = num_channels_projection
        self.num_channels_inv_projection = num_channels_inv_projection
        self.mode = mode

        self.inv_transform_2d = ResNet2d(self.img_shape, channels_2d,
                                         strides_2d)

        input_shape = self.inv_transform_2d.output_shape
        self.inv_projection = InverseProjection(input_shape, num_channels_inv_projection)


        self.inv_transform_3d = ResNet3d(self.inv_projection.output_shape,
                                         channels_3d, strides_3d)
        self.rotation_layer = Rotate3d(self.mode)


        forward_channels_3d = list(reversed(channels_3d))[1:] + [channels_3d[0]]
        forward_strides_3d = [-stride if abs(stride) == 2 else 1 for stride in list(reversed(strides_3d[1:]))] + [strides_3d[0]]
        self.transform_3d = ResNet3d(self.inv_transform_3d.output_shape,
                                     forward_channels_3d, forward_strides_3d)

        self.projection = Projection(self.transform_3d.output_shape,
                                     num_channels_projection)

        forward_channels_2d = list(reversed(channels_2d))[1:] + [channels_2d[0]]
        forward_strides_2d = [-stride if abs(stride) == 2 else 1 for stride in list(reversed(strides_2d[1:]))] + [strides_2d[0]]
        final_conv_channels_2d = img_shape[0]
        self.transform_2d = ResNet2d(self.projection.output_shape,
                                     forward_channels_2d, forward_strides_2d,
                                     final_conv_channels_2d)

        self.scene_shape = self.inv_transform_3d.output_shape
        self.spherical_mask = SphericalMask(self.scene_shape)


    def decode(self, scene):

        features_3d = self.transform_3d(scene)
        features_2d = self.projection(features_3d)
        return torch.sigmoid(self.transform_2d(features_2d))

    def encode(self, img):

        features_2d = self.inv_transform_2d(img)
        features_3d = self.inv_projection(features_2d)
        scene = self.inv_transform_3d(features_3d)
        return self.spherical_mask(scene)

    def encode_pose(self, img):
        features_2d = self.inv_transform_2d(img)
        features_3d = self.inv_projection(features_2d)
        scene = self.inv_transform_3d(features_3d)
        return self.spherical_mask(scene), None

    def act(self, z, action, *args):
        #The matrix is processed by inverting, flipping rows and columns and adding a
        #column of zeros as the translational part
        matrix = torch.flip(action.transpose(-2, -1), dims=(-2, -1))
        translations = torch.zeros(matrix.shape[0], 3, 1, device=matrix.device)
        affine_matrix = torch.cat([matrix, translations], dim=2)

        #Because of some PyTorch mystery, align_corners should be False
        grid = F.affine_grid(affine_matrix, z.shape, align_corners=False)
        z_rotated = F.grid_sample(z, grid, align_corners=False)
        return z_rotated

    def act_k(self, z, action, action_type, group_dim, method):
        for i in range(action.shape[1]):
            z = self.act(z, action[:, i, ...])
        return z


    def forward(self, x, action):
        z = self.encode(x)
        z_rotated = self.act(z, action)
        return self.decode(z_rotated), self.decode(z)

if __name__ == "__main__":
    pass
