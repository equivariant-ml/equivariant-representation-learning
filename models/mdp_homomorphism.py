import torch
import torch.nn as nn
import ipdb
from models.models_nn import BaseDecoder, BaseEncoder
from utils.nn_utils import normalize
from models.resnet import ResNet18Dec, ResNet18Enc


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class MDPHomomorphism(nn.Module):
    def __init__(self, nc, latent_dim, action_dim, action_type, regularization, model):
        super().__init__()



        if model == 'mdp_resnet':
            self.encoder = ResNet18Enc(z_dim=latent_dim, nc=nc)
            self.decoder = ResNet18Dec(z_dim=latent_dim, nc=nc)
        else:
            self.encoder = BaseEncoder(nc, latent_dim)
            self.decoder = BaseDecoder(nc, latent_dim)

        hidden = 128
        self.f_out = nn.Sequential(nn.Linear(latent_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim))

        self.action_type = action_type
        self.action_dim = action_dim

        self.regularization = regularization

    def encode_pose(self, x):
        if self.regularization == 'info-nce':
            return normalize(self.encoder(x)), None
        return self.encoder(x), None

    def decode(self, z):
        return self.decoder(z)

    def act(self, z_encoded, action, *args):
        action = action.view(action.shape[0], -1)
        return self.f_out(torch.cat((z_encoded, action), -1))

    def act_k(self, z_encoded, action, action_type, group_dim, method):
        for i in range(action.shape[1]):
            z_encoded = self.act(z_encoded, action[:,i,:])
        return z_encoded

    def forward(self, x):
        return None, None, self.encode_pose(x)
