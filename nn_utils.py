import torch
import torch.nn.functional as F
import numpy as np
import os
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
import matplotlib.pyplot as plt


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def entropy_loss(recon_x, x):
    return F.binary_cross_entropy(recon_x, x, reduction = 'mean')

def normalize(z, EPS=1e-8):
    norm = torch.norm(z, p=2, dim=-1, keepdim=True) #+ EPS
    return z.div(norm)

def quaternion_distance(q1, q2):
    dist1 = torch.norm(q1-q2, p=2, dim = -1)
    dist2 = torch.norm(q1 + q2, p=2, dim=-1)
    dist = torch.min(dist1, dist2).mean()

    return dist

def equivariance_loss(model, z, action, action_dim, img_next, action_type, method='quaternion', device='cuda'):
    true_pose = model.encode_pose(img_next)

    if action_type == 'translate':
        z_out = torch.tanh(z[:, :action_dim])
        pose = z_out + action
        equiv_loss = F.mse_loss(pose, true_pose)
    elif action_type == 'rotate':
        if method == 'naive':
            """
            Encode naively to R^3 and multiply by matrix
            """
            pose = (z[:, None, :action_dim] @ action).squeeze(1)
            equiv_loss = ((pose - true_pose)**2).mean()
        elif method == 'quaternion':
            z_quaternion = normalize(z[:, :action_dim])
            pose_to_matrix = quaternion_to_matrix(z_quaternion)
            pose = pose_to_matrix @ action
            pose_to_quaternion = matrix_to_quaternion(pose)
            equiv_loss = quaternion_distance(pose_to_quaternion, true_pose)


    elif action_type ==  'isometries_2d':
        z_out = model.pose_activation(z[:, :action_dim])
        trans_loss =  ((z_out[:, :2] +   action[:, :2] - true_pose[:, :2])**2).mean()
        angle_loss = (torch.abs( torch.complex(z_out[:, 2], z_out[:, 3])*torch.complex(torch.cos(action[:, 2]), torch.sin(action[:, 2])) - torch.complex(true_pose[:, 2], true_pose[:, 3]))**2).mean()
        equiv_loss = trans_loss + angle_loss

    return equiv_loss
