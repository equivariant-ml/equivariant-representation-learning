from lib2to3.pytree import Base
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
import utils.nn_utils as nn_utils
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
import ipdb

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class AE(nn.Module):
    def __init__(self, encoder, decoder, latent_dim, action_dim, encoding, regularization, device='cuda'):
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.action_dim = action_dim

        self.encoding = encoding
        self.device = device
        self.regularization = regularization

    def encode(self, x):
        encoded = self.encoder(x)
        return encoded

    def decode(self, z):
        return self.decoder(z)

    def encode_pose(self, x):
        """
        encodes image x -> G x E

        """
        z = self.encode(x)
        z_pose = self.pose_activation(z[:, :self.action_dim])
        z_extra = z[:, self.action_dim:]

        if self.regularization == 'info-nce':
            z_extra = nn_utils.normalize(z_extra)

        return torch.cat((z_pose, z_extra), -1), z

    def pose_activation(self, z):
        # ipdb.set_trace()
        if self.encoding == 'tanh':
            return torch.tanh(z)
        elif self.encoding == 'normalize':
            return nn_utils.normalize(z)
        elif self.encoding == 'non':
            return z
        elif self.encoding == 'trans_angle':
            return  torch.cat((z[:, :2], torch.cos(z[:,2]).unsqueeze(1), torch.sin(z[:,2]).unsqueeze(1)), -1)
        elif self.encoding == 'lie_rotation':
            return torch.matrix_exp(nn_utils.antisym_matrix(z)).view(-1, 9)

    def act(self, pose_plus_extra, action, action_type, group_dim, method):
        batch_size = pose_plus_extra.shape[0]
        pose = pose_plus_extra[:, :group_dim]
        extra = pose_plus_extra[:, group_dim:]
        if action_type == 'translate':
            new_pose = pose + action
        elif action_type == 'rotate':
            if method == 'naive':
                new_pose = (action @ pose.unsqueeze(-1)).squeeze(-1)
            elif method == 'lie':
                new_pose = action @ pose.view((batch_size, 3, 3))
                new_pose = new_pose.view(batch_size, -1)
            elif method == 'lie_right':
                new_pose = pose.view((batch_size, 3, 3)) @ action
                new_pose = new_pose.view(batch_size, -1)
            elif method == 'quaternion':
                pose_to_matrix = quaternion_to_matrix(pose)
                pose = action @ pose_to_matrix
                pose_to_quaternion = matrix_to_quaternion(pose)
                new_pose = pose_to_quaternion
        elif action_type == 'isometries_2d':
            res = torch.complex(pose[:,2], pose[:,3]) * torch.complex(torch.cos(action[:,2]), torch.sin(action[:,2]))
            new_pose = torch.cat((pose[:, :2] + action[:, :2], res.real.unsqueeze(1), res.imag.unsqueeze(1)), dim=-1)
        elif action_type == 'isometries_2d_local':
            new_trans = torch.complex(pose[:, 0], pose[:, 1]) + torch.complex(pose[:, 2], pose[:, 3]) * torch.complex(action[:, 0], action[:, 1])
            new_rot = torch.complex(pose[:,2], pose[:,3]) * torch.complex(torch.cos(action[:,2]), torch.sin(action[:,2]))
            new_pose = torch.cat((new_trans.real.unsqueeze(1), new_trans.imag.unsqueeze(1), new_rot.real.unsqueeze(1), new_rot.imag.unsqueeze(1)), dim=-1)

        new_z_encoded = torch.cat((new_pose, extra), -1)
        return new_z_encoded


    def act_k(self, pose, action, action_type, group_dim, method):
        for i in range(action.shape[1]):
            pose = self.act(pose, action[:,i,...], action_type, group_dim, method)
        return pose



class BaseEncoder(nn.Module):
    def __init__(self, nc, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
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

    def forward(self, x):
        return self.encoder(x)


class BaseDecoder(nn.Module):
    def __init__(self, nc, latent_dim):
        super().__init__()
        self.decoder = nn.Sequential(
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

    def forward(self, x):
        return self.decoder(x)



class AE_CNN_Lie(AE):                           #Works on 64x64 images
    def __init__(self, extra_dim, nc, action_dim, group_dim, encoding, regularization):

        latent_dim = extra_dim + action_dim
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
            nn.Linear(extra_dim + group_dim, 256),
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
        super(AE_CNN_Lie, self).__init__(encoder, decoder, latent_dim, action_dim, encoding, regularization)




if __name__ == "__main__":
    pass
