from multiprocessing.sharedctypes import Value
from tkinter import E
from comet_ml import Experiment
from collections import defaultdict

import torch
import torch.utils.data
from torch import nn, optim, save, load
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchvision import transforms
import numpy as np
from operator import mul
from functools import reduce
import os
import matplotlib.pyplot as plt
import argparse
import pickle
from datasets.custom_dset import CustomDataset
from datasets.neural_render_dset import ShapenetDataset

from models.models_nn import *
from utils.nn_utils import *
from utils.parse_args import get_args
from models.resnet import ResnetV2, ResnetV2_Lie
from models.mdp_homomorphism import MDPHomomorphism

# Import datasets
from datasets.platonic_dset import PlatonicMerged
from datasets.equiv_dset import *
from datasets.custom_dset import CustomMerged, CustomMergedTrajectory

from ENR.models_ENR import NeuralRenderer
from utils.utils import CometDummy, get_hinge_loss
from torchvision.utils import make_grid


torch.cuda.empty_cache()

parser = get_args()
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device: ', device)

if args.use_comet:
    e = Experiment(project_name=os.env['COMET_PROJECT_NAME'], workspace=os.env['COMET_WORKSPACE'],
            api_key=os.env['COMET_API_KEY'])
else:
    e = CometDummy()

e.log_parameters(vars(args))

# Save paths
MODEL_PATH = os.path.join(args.checkpoints_dir, args.model_name)

figures_dir = os.path.join(MODEL_PATH, 'figures')
model_file = os.path.join(MODEL_PATH, 'model.pt')
meta_file = os.path.join(MODEL_PATH, 'metadata.pkl')
log_file = os.path.join(MODEL_PATH, 'log.txt')

make_dir(MODEL_PATH)
make_dir(figures_dir)

pickle.dump({'args': args}, open(meta_file, 'wb'))


# Set dataset
if args.dataset == 'sprites':
    ACTION_TYPE = 'translate'
    encoding = 'non'
    action_dim = 3
    group_dim = 3
    dset = EquivDataset(f'{args.data_dir}/sprites_data/', greyscale=True)
    dset_10 = EquivDataset(f'{args.data_dir}/sprites_data/', greyscale=True, length_trajectory=10)
    dset_20 = EquivDataset(f'{args.data_dir}/sprites_data/', greyscale=True, length_trajectory=20)
elif args.dataset == 'color-shift':
    ACTION_TYPE = 'translate'
    encoding = 'non'
    action_dim = 3
    group_dim = 3
    dset = EquivDataset(f'{args.data_dir}/colorshift_data/')
    dset_10 = EquivDataset(f'{args.data_dir}/colorshift_data/', length_trajectory=10)
    dset_20 = EquivDataset(f'{args.data_dir}/colorshift_data/', length_trajectory=20)
elif args.dataset == 'multi-sprites':
    ACTION_TYPE = 'translate'
    encoding = 'non'
    action_dim = 6
    group_dim = 6
    dset = EquivDataset(f'{args.data_dir}/multisprites_data/')
    dset_10 = EquivDataset(f'{args.data_dir}/multisprites_data/', length_trajectory=10)
    dset_20 = EquivDataset(f'{args.data_dir}/multisprites_data/', length_trajectory=20)
elif args.dataset == 'chairs':
    ACTION_TYPE = 'rotate'
    if args.method == 'naive':
        encoding = 'non'
        action_dim = 3
        group_dim = 3
    elif args.method == 'quaternion':
        encoding = 'normalize'
        action_dim = 4
        group_dim = 4
    elif args.method == 'lie':
        encoding = 'lie_rotation'
        action_dim = 3
        group_dim = 9
    elif args.method == 'enr':
        action_dim = 3
        group_dim = 9
    elif args.method == 'mdp':
        action_dim = 9
        group_dim = 9
    dset = CustomMerged(data_dir=args.data_dir, dataset=args.dataset)
    dset_10 = CustomMergedTrajectory(data_dir=args.data_dir, dataset=args.dataset, trajectory_length=10)
    dset_20 = CustomMergedTrajectory(data_dir=args.data_dir, dataset=args.dataset, trajectory_length=20)
elif args.dataset == 'room_combined':
    ACTION_TYPE = 'isometries_2d'
    encoding = 'trans_angle'
    action_dim = 3
    group_dim = 4
    dset = EquivDataset(f'{args.data_dir}/gibson_global_frame/')
    dset_10 = EquivDataset(f'{args.data_dir}/gibson_global_frame/', length_trajectory=10)
    dset_20 = EquivDataset(f'{args.data_dir}/gibson_global_frame/', length_trajectory=20)
else:
    print("Invalid dataset")

extra_dim = args.extra_dim
latent_dim = action_dim + extra_dim

dset, dset_test = torch.utils.data.random_split(dset, [len(dset) - int(len(dset)/10), int(len(dset)/10)])
train_loader = torch.utils.data.DataLoader(dset,
                                           batch_size=args.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dset_test,
                                            batch_size=32, shuffle=True)   #hard-coded validation batch-size for a fair hit-rate
traj_loader_10 = torch.utils.data.DataLoader(dset_10,
                                            batch_size=32, shuffle=True)
traj_loader_20 = torch.utils.data.DataLoader(dset_20,
                                            batch_size=32, shuffle=True)

print("# train set:", len(dset))
print("# test set:", len(dset_test))

# Sample data
img, _, acn, _ = next(iter(train_loader))
img_shape = img.shape[1:]
print("Dataset min/max", img.min(), img.max())


if args.model == 'cnn':
    model = AE_CNN_Lie(extra_dim, img_shape[0], action_dim, group_dim, encoding, args.regularization).to(device)
elif args.model == 'resnet_lie':
    model = ResnetV2_Lie(extra_dim, img_shape[0], action_dim, group_dim, encoding, args.regularization).to(device)
elif args.model == 'enr':
    model = NeuralRenderer([3, 64, 64], [64, 64, 128, 128, 128, 128, 256, 256, 128, 128, 128],
                        [1, 1, 2, 1, 2, 1, 2, 1, -2, 1, 1], [32, 32, 128, 128, 128, 64, 64, 64],
                        [1, 1, 2, 1, 1, -2, 1, 1], [256, 512, 1024],  [512, 256, 256]).to(device)
elif args.model == 'mdp' or args.model == 'mdp_resnet':
    model = MDPHomomorphism(img_shape[0], extra_dim, reduce(mul, acn.shape[1:], 1), ACTION_TYPE, args.regularization, args.model).to(device)
else:
    raise ValueError(f"model not specified {args.model}")



optimizer = optim.Adam(model.parameters(), lr=args.lr)

errors = defaultdict(list)

def train(epoch, data_loader, mode='train'):

    mu_loss = 0
    mu_hitrate = 0
    mu_equiv_loss = 0
    mu_reg = 0

    global_step = len(data_loader) * epoch

    for batch_idx, (img, img_next, action, _) in enumerate(data_loader):
        batch_size = img.shape[0]
        global_step += batch_size

        if mode == 'train':
            optimizer.zero_grad()
            model.train()
        elif mode == 'val':
            model.eval()

        loss = torch.Tensor([0]).to(device)
        reg = torch.Tensor([0]).to(device)
        equiv_loss = torch.Tensor([0]).to(device)
        total_recon_loss = torch.Tensor([0]).to(device)

        img = img.to(device)
        img_next = img_next.to(device)
        action = action.to(device)

        # Encode: X -> G x E
        z_encoded, z_pre = model.encode_pose(img)
        z_next_encoded, z_next_pre = model.encode_pose(img_next)
        z_next_pred = model.act(z_encoded, action, ACTION_TYPE, group_dim, args.method)

        equiv_loss = clean_equivariance_loss(z_next_pred, z_next_encoded, z_pre, z_next_pre, action, action_dim, ACTION_TYPE, args.method, args)

        mu_equiv_loss += equiv_loss.item()
        loss += equiv_loss

        if args.regularization == 'hinge':
            reg = get_hinge_loss(z_next_pred, z_next_encoded, group_dim, args.model, args.method, args.decoder)
            loss += reg
        elif args.regularization == 'info-nce':
            reg = infoNCE(z_next_pred, z_next_encoded, group_dim, tau=args.tau)
            loss += reg
        elif args.regularization == 'none':
            pass

        mu_reg += reg.item()

        if args.decoder:
            recon_1 = model.decode(z_encoded)
            recon_2 = model.decode(z_next_encoded)
            total_recon_loss = 0.5 * (F.binary_cross_entropy(recon_1, img) + F.binary_cross_entropy(recon_2, img_next))
            loss += total_recon_loss
            mu_loss += total_recon_loss.item()
            e.log_metric(f"recon_{mode}", total_recon_loss, step=global_step, epoch=epoch)

        ## Statistics
        hitrate = hitRate(z_next_pred, z_next_encoded, group_dim, args.regularization, args.method)
        mu_hitrate += hitrate.item()

        if args.model == 'enr':
            loss, total_recon_loss, recon_1 = get_enr_loss(args, model, img, img_next, action)
            equiv_loss = torch.Tensor([0])
            e.log_metric(f"recon_{mode}", total_recon_loss, step=global_step, epoch=epoch)

        print_string = f"{mode.upper()} Epoch: {epoch}, Batch: {batch_idx} / {len(data_loader)} \
                    Equiv: {equiv_loss.item():.3} {args.regularization} {reg.item():.3} Hitrate: {hitrate:.3f}"


        if mode == 'train':
            loss.backward()
            optimizer.step()

        if batch_idx % args.log_interval == 0 and (mode == 'train' or mode == 'val'):
            print(print_string)

        e.log_metric(f"loss_{mode}", loss, step=global_step, epoch=epoch)
        e.log_metric(f"equiv_{mode}", equiv_loss, step=global_step, epoch=epoch)
        e.log_metric(f"hitrate_{mode}", hitrate, step=global_step, epoch=epoch)

    mu_loss /= len(data_loader)
    mu_hitrate /= len(data_loader)
    mu_equiv_loss /= len(data_loader)
    mu_reg /= len(data_loader)


    if mode == 'val':
        mu_traj_10 = 0
        mu_traj_20 = 0

        with torch.no_grad():
            mu_traj_10, _ = evaluate_trajectory(args, model, ACTION_TYPE, group_dim, traj_loader_10, e, device, mode, global_step, epoch, decoder=args.decoder)
            mu_traj_20, _ = evaluate_trajectory(args, model, ACTION_TYPE, group_dim, traj_loader_20, e, device, mode, global_step, epoch, decoder=args.decoder)

            e.log_metric(f'hitrate_traj_10_{mode}', mu_traj_10, step=global_step, epoch=epoch)
            e.log_metric(f'hitrate_traj_20_{mode}', mu_traj_20, step=global_step, epoch=epoch)

        print(f"{mode.upper()} Epoch: {epoch}, Loss: {mu_loss:.3} Mu Hitrate: {mu_hitrate:.3} Traj Hitrate 10 {mu_traj_10} Traj Hitrate 20 {mu_traj_20}")
        if args.decoder or args.model == 'enr':
            save_image(recon_1[:16], f'{figures_dir}/recon_{str(epoch)}.png')
            grid = make_grid(recon_1[:16])
            # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            fig, ax = plt.subplots()
            ax.imshow(np.asarray(im))
            e.log_figure(figure=fig)
            plt.close('all')
            plt.cla()
            plt.clf()

        errors['mu_loss'].append(mu_loss)
        errors['mu_hitrate'].append(mu_hitrate)
        errors['mu_equiv_loss'].append(mu_equiv_loss)
        errors['mu_reg'].append(mu_reg)
        errors['mu_traj_10'].append(mu_traj_10)
        errors['mu_traj_20'].append(mu_traj_20)

        filename = f'{MODEL_PATH}/errors_val.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(errors, f)

        if (epoch % args.save_interval) == 0:
            save(model, model_file)



if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch, train_loader, 'train')
        with torch.no_grad():
            train(epoch, val_loader, 'val')
