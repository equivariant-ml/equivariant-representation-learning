import torch
import torch.nn.functional as F
import numpy as np
import os
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix, so3_log_map
import matplotlib.pyplot as plt
from scipy.linalg import logm
import ipdb
from tqdm import tqdm

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def entropy_loss(recon_x, x):
    return F.binary_cross_entropy(recon_x, x, reduction = 'mean')

def normalize(z, EPS=1e-8):
    norm = torch.norm(z, p=2, dim=-1, keepdim=True) #+ EPS
    return z.div(norm)

def quaternion_distance(q1, q2, un_meaned=False):
    dist1 = torch.norm(q1-q2, p=2, dim = -1)
    dist2 = torch.norm(q1 + q2, p=2, dim=-1)
    dist = torch.min(dist1, dist2)

    if not un_meaned:
        dist = dist.mean()

    return dist

def clean_equivariance_loss(z_next_pred, z_next_true, z_pre, z_next_pre, action, action_dim ,action_type, method, args):
    # ipdb.set_trace()
    if args.model == 'enr':
        return torch.Tensor([0]).to(z_next_pred.device)
    if args.model in ['mdp', 'mdp_resnet']:
        return F.mse_loss(z_next_pred, z_next_true)

    if action_type == 'translate':
        equiv_loss = F.mse_loss(z_next_pred[:, :action_dim], z_next_true[:, :action_dim])
    elif action_type == 'rotate':
        if method == 'naive':
            equiv_loss = F.mse_loss(z_next_pred[:, :action_dim], z_next_true[:, :action_dim])
        elif method == 'quaternion':
            equiv_loss = quaternion_distance(z_next_pred[:, :action_dim], z_next_true[:, :action_dim])
        elif method == 'lie_right':
            z_pose = antisym_matrix(z_pre[:, :action_dim])
            z_pose_next = antisym_matrix(z_next_pre[:, :action_dim])
            z1 = torch.matrix_exp(z_pose)
            z2 = torch.matrix_exp(z_pose_next)
            equiv_loss = torch.linalg.matrix_norm(z1 @ action - z2, ord='nuc').mean()
        elif method == 'lie':
            z_pose = antisym_matrix(z_pre[:, :action_dim])
            z_pose_next = antisym_matrix(z_next_pre[:, :action_dim])
            z1 = torch.matrix_exp(z_pose)
            z2 = torch.matrix_exp(z_pose_next)
            equiv_loss = F.mse_loss(action @ z1, z2)
    elif action_type ==  'isometries_2d':
        trans_loss = F.mse_loss(z_pre[:, :2] + action[:, :2], z_next_pre[:, :2])
        BKH = z_next_pre[:, 2] - z_pre[:, 2]
        angle_loss = torch.abs(torch.complex(torch.cos(BKH), torch.sin(BKH)) - torch.complex(torch.cos(action[:, 2]), torch.sin(action[:, 2]))).mean()
        equiv_loss = trans_loss + angle_loss
    elif action_type ==  'isometries_2d_local':
        trans_loss = (torch.abs(torch.complex(z_pre[:, 0], z_pre[:, 1]) + torch.complex(z_pre[:, 2], z_pre[:, 3]) * torch.complex(action[:, 0], action[:, 1]) - torch.complex(z_next_pre[:, 0], z_next_pre[:, 1]))**2).mean()
        BKH = z_next_pre[:, 2] - z_pre[:, 2]
        angle_loss = (torch.abs(torch.complex(torch.cos(BKH), torch.sin(BKH)) - torch.complex(torch.cos(action[:, 2]), torch.sin(action[:, 2])))**2).mean()
        equiv_loss = trans_loss + angle_loss
    else:
        equiv_loss = torch.Tensor([0]).to(z_next_pred.device)

    return equiv_loss


def get_equivariance_loss(args, model, img, img_next, action, action_dim, ACTION_TYPE, device):
    if args.model == 'mdp':
        z = model.encode(img)
        z_next_pred = model.act(z, action)
        z_next_true = model.encode(img_next)
        equiv_loss = F.mse_loss(z_next_pred, z_next_true)

        z2 = z_next_true
        z2_rand = z2[torch.randperm(len(z2))]
        distance = torch.norm(z2 - z2_rand, p=2, dim=-1)

        entropy_reg = torch.max(torch.zeros_like(distance).to(device), torch.ones_like(distance).to(device) - distance).mean()

        loss = equiv_loss + entropy_reg
        total_recon_loss = torch.Tensor([0]).to(img.device)
        recon = torch.Tensor([0]).to(img.device)
    else:
        recon, z, _ = model(img)
        recon_next, z_next, _ = model(img_next)

        recon_loss = entropy_loss(recon, img)
        recon_loss_next = entropy_loss(recon_next, img_next)

        total_recon_loss = 0.5 * (recon_loss + recon_loss_next)

        equiv_loss = equivariance_loss(model, z, action, action_dim, img_next, action_type=ACTION_TYPE, device=device, method=args.method)
        ce_loss = torch.sum((z[:, action_dim : ] - z_next[:, action_dim:])**2, -1).mean(0)

        # Hinge-loss
        z2 = z_next[:, action_dim : ]
        z2_rand = z2[torch.randperm(len(z2))]
        distance = torch.norm(z2 - z2_rand, p=2, dim=-1)

        entropy_reg = torch.max(torch.zeros_like(distance).to(device), torch.ones_like(distance).to(device) - distance).mean()

        #mu_loss += total_recon_loss.detach().cpu().numpy()

        contra_loss = infoNCE(z[:,action_dim:], z_next[:, action_dim:], 0.05)

        if args.decoder:
            #loss = 0.1*equiv_loss  +  ce_loss + total_recon_loss
            loss = equiv_loss + contra_loss + total_recon_loss
        else:
            loss = equiv_loss +  ce_loss + entropy_reg

    return loss, total_recon_loss, equiv_loss, entropy_reg, recon

def get_enr_loss(args, model, img, img_next, action):
    equiv_recon, recon = model(img, action)

    loss = entropy_loss(equiv_recon, img_next)

    with torch.no_grad():
        recon_next = model.decode(model.encode(img_next))
        total_recon_loss = 0.5 * (entropy_loss(recon, img) + entropy_loss(recon_next, img_next))

    return loss, total_recon_loss, recon



def hitRateUnnormalized(z, z_next, action, device, action_dim, action_type, method, model):
    # extra = normalize(z[:,action_dim:])
    # extra_next = normalize(z_next[:,action_dim:])

    extra = z[:, action_dim:]
    extra_next = z_next[:, action_dim:]

    pose = model.pose_activation(z[:,:action_dim])
    pose_next = model.pose_activation(z_next[:,:action_dim]).view((len(z_next), -1))

    pose_pred = model.act(pose, action, action_type, method).view((len(pose), -1))

    pred = torch.cat((pose_pred, extra), dim=-1)
    true = torch.cat((pose_next, extra_next), dim=-1)

    distance_matrix = ((pred.unsqueeze(0) - true.unsqueeze(1))**2).sum(-1)
    _, idxs = torch.min(distance_matrix, dim=0)
    ordered = torch.arange(start=0, end=len(z_next), step=1).to(device)
    return torch.eq(idxs, ordered).double().mean()

def hitRate(z_pred, z_true, group_dim, regularization, method):
    # ipdb.set_trace()
    z_pred = torch.flatten(z_pred, 1) # batch_dim * 64*16*16*16
    z_true = torch.flatten(z_true, 1)
    batch_size = z_pred.shape[0]
    device = z_pred.device

    distance_matrix_pose = ((z_pred[:, :group_dim].unsqueeze(0) - z_true[:, :group_dim].unsqueeze(1))**2).sum(-1)
    distance_matrix_class = ((z_pred[:, group_dim:].unsqueeze(0) - z_true[:, group_dim:].unsqueeze(1))**2).sum(-1)

    if method == 'lie':
        if group_dim == 9:
            z_pred_matrix = z_pred[:, :group_dim].view(1, batch_size, 3, 3)
            z_true_matrix = z_true[:, :group_dim].view(batch_size, 1, 3, 3)
            distance_matrix_pose = torch.linalg.matrix_norm(z_pred_matrix - z_true_matrix, ord='fro')**2


    if regularization == 'info-nce' and method != 'mdp':
        distance_matrix_class = 1.0 - (z_pred[:, group_dim:].unsqueeze(0) * z_true[:, group_dim:].unsqueeze(1)).sum(-1)

    distance_matrix = 1.0 * distance_matrix_pose + 1.0 * distance_matrix_class
    if method == 'lie':
        distance_matrix = 1.0 * distance_matrix_pose + 1.0 * distance_matrix_class


    _, idxs = torch.min(distance_matrix, dim=0)
    ordered = torch.arange(start=0, end=batch_size, step=1).to(device)
    return torch.eq(idxs, ordered).double().mean()


def antisym_matrix(z):
    res = torch.zeros((z.shape[0], 3, 3)).to(z.device)
    res[:,0,1 ] = z[:, 0]
    res[:,0,2 ] = z[:, 1]
    res[:,1, 0 ] = -z[:, 0]
    res[:,1,2 ] = z[:, 2]
    res[:,2,0 ] = -z[:, 1]
    res[:,2,1 ] = -z[:, 2]
    return res

def bracket(a,b):
    return a@b - b@a


def infoNCE(z_pred, z_next, group_dim, tau):
    batch_size = len(z_next)
    extra = z_pred[:, group_dim:]
    extra_next = z_next[:, group_dim:]
    distance_matrix = torch.exp((extra.unsqueeze(1) * extra_next.unsqueeze(0)).sum(-1) / tau)
    #denominator = torch.diagonal(distance_matrix @ (1. - torch.eye(batch_size)).to(z_pred.device))
    denominator = distance_matrix.sum(-1)  #this version does not exclude the diagonal indices
    return - torch.mean((extra * extra_next).sum(-1) / tau - torch.log(denominator))



def evaluate_trajectory(args, model, ACTION_TYPE, group_dim, traj_loader, e, device, mode, global_step, epoch, decoder=False):
    mu_hitrate_traj = 0

    mu_recon_traj = 0


    for batch_idx, (img, img_next, action, _) in enumerate(traj_loader):
        img = img.to(device)
        img_next = img_next.to(device)
        action = action.to(device)

        z_encoded, z_pre = model.encode_pose(img)
        z_next_encoded, z_next_pre = model.encode_pose(img_next)
        z_next_pred = model.act_k(z_encoded, action, ACTION_TYPE, group_dim, args.method)

        if decoder == True:
            decoded_k = model.decode(z_next_pred)
            recon = F.binary_cross_entropy(decoded_k, img_next)
            mu_recon_traj += recon.item()

        hitrate = hitRate(z_next_pred, z_next_encoded, group_dim, args.regularization, args.method)
        mu_hitrate_traj += hitrate.item()
    mu_hitrate_traj /= len(traj_loader)
    return mu_hitrate_traj, mu_recon_traj


def evaluate_trajectory_recon(args, representation_learner, decoder, ACTION_TYPE, group_dim, traj_loader, device):
    with torch.no_grad():
        mu_recon_traj = 0
        mu_hitrate_traj = 0
        for batch_idx, (img, img_next, action, _) in tqdm(enumerate(traj_loader), total=len(traj_loader)):
            img = img.to(device)
            img_next = img_next.to(device)
            action = action.to(device)

            z_encoded, z_pre = representation_learner.encode_pose(img)
            z_next_encoded, z_next_pre = representation_learner.encode_pose(img_next)
            z_next_pred = representation_learner.act_k(z_encoded, action, ACTION_TYPE, group_dim, args.method)

            decoded_k = decoder.decode(z_next_pred)
            recon = F.binary_cross_entropy(decoded_k, img_next)
            mu_recon_traj += recon.item()

            hitrate = hitRate(z_next_pred, z_next_encoded, group_dim, args.regularization, args.method)
            mu_hitrate_traj += hitrate.item()

        mu_hitrate_traj /= len(traj_loader)
        mu_recon_traj /= len(traj_loader)

        return mu_hitrate_traj, mu_recon_traj
