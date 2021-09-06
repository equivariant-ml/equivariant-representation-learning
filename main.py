import torch
import torch.utils.data
from torch import nn, optim, save, load
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchvision import transforms
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import pickle

from models_nn import *
from nn_utils import *
from resnet import ResnetV2

# Import datasets
from datasets.platonic_dset import PlatonicMerged
from datasets.equiv_dset import *


torch.cuda.empty_cache()

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42, help="Set seed for training")

# Training details
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--log-interval', type=int, default=10)
parser.add_argument('--val-interval', type=int, default=10)
parser.add_argument('--save-interval', default=10, type=int, help="Epoch to save model")

# Dataset
parser.add_argument('--dataset', default='sprites', type=str, help="Dataset")
parser.add_argument('--batch-size', type=int, default=64, help="Batch size")


parser.add_argument('--action-dim', default=2, type=int, help="Dimension of the group")
parser.add_argument('--extra-dim', default=3, type=int, help="Number of classes")
parser.add_argument('--model-name', required=True, type=str, help="Name of model")
parser.add_argument('--decoder', action='store_true', default=True, help="Do you need a decoder?")
parser.add_argument('--model', default='resnet', type=str, help="Model to use")

# Rotation specific arguments
parser.add_argument('--method', type=str, default='quaternion', help="What loss to use for rotations")

parser.add_argument('--checkpoints-dir', default='checkpoints', type=str)

# Optimization
parser.add_argument('--lr-scheduler', action='store_true', default=False, help="Use a lr scheduler")
parser.add_argument('--lr', default=1e-3, type=float)

parser.add_argument('--data-dir', default='data', type=str)

args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.autograd.detect_anomaly()

# Save paths
MODEL_PATH = os.path.join(args.checkpoints_dir, args.model_name)

figures_dir = os.path.join(MODEL_PATH, 'figures')
model_file = os.path.join(MODEL_PATH, 'model.pt')
meta_file = os.path.join(MODEL_PATH, 'metadata.pkl')
log_file = os.path.join(MODEL_PATH, 'log.txt')

make_dir(MODEL_PATH)
make_dir(figures_dir)

pickle.dump({'args': args}, open(meta_file, 'wb'))

# Set dimensions
action_dim = args.action_dim
extra_dim = args.extra_dim           #Also number of classes
latent_dim = action_dim + extra_dim



# Set group action
if args.dataset == 'sprites' or args.dataset == 'color-shift' or args.dataset == 'multi-sprites':
    ACTION_TYPE = 'translate'
    encoding = 'tanh'
if args.dataset == 'platonics':
    ACTION_TYPE = 'rotate'
    if args.method == 'naive':
        encoding = 'non'
    elif args.method == 'quaternion':
        encoding = 'normalize'
if args.dataset == 'room_combined':
    ACTION_TYPE = 'isometries_2d'
    encoding = 'trans_angle'


# Allocate datasetPlatonic
if args.dataset == 'color-shift':
    dset = EquivDataset('data/colorshift_data/')
elif args.dataset == 'sprites':
    dset = EquivDataset('data/sprites_data/', greyscale = True)
elif args.dataset == 'multi-sprites':
    dset = EquivDataset('data/multisprites_data/')
elif args.dataset == 'platonics':
    dset = PlatonicMerged(30000, big=True, data_dir=args.data_dir)
elif args.dataset == 'room_combined':
    dset = EquivImgDataset('./data/combine/', combined=True)
else:
    print("Invalid dataset")


train_data, valid_data = torch.utils.data.random_split(dset, [len(dset) - int(len(dset)/10), int(len(dset)/10)])
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=args.batch_size, shuffle = True)
val_loader = torch.utils.data.DataLoader(valid_data,
                                           batch_size=args.batch_size, shuffle = True)

# Sample data
img, _, _, _ = next(iter(train_loader))
img_shape = img.shape[1:]


# Create Model
if args.model == 'cnn':
    model = AE_CNN(latent_dim, img_shape[0], args.action_dim, encoding).to(device)
elif args.model == 'resnet':
    model = ResnetV2(latent_dim, img_shape[0], args.action_dim, encoding).to(device)


optimizer = optim.Adam(model.parameters(), lr=args.lr)

if args.lr_scheduler:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20)

errors = []

def train(epoch, data_loader, mode='train'):

    mu_loss = 0

    global_step = len(data_loader) * epoch

    for batch_idx, (img, img_next, action, classes) in enumerate(data_loader):
        global_step += 1

        if mode == 'train':
            optimizer.zero_grad()
            model.train()
        elif mode == 'val':
            model.eval()

        img = img.to(device)
        img_next = img_next.to(device)
        action = action.to(device)
        # if args.dataset == 'multi-sprites':
        #     action = action.squeeze(1)


        recon, z = model(img)
        recon_next, z_next = model(img_next)

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

        mu_loss += total_recon_loss.detach().cpu().numpy()

        if args.decoder == True:
            loss = equiv_loss +  ce_loss + total_recon_loss
        else:
            loss = equiv_loss +  ce_loss + entropy_reg



        if mode == 'train':
            loss.backward()
            optimizer.step()

        if batch_idx % args.log_interval == 0 and mode == 'train':
            print(f"{mode.upper()} Epoch: {epoch}, Batch: {batch_idx} / {len(train_loader)} \
                    Recon Loss: {total_recon_loss.item():.3} Equiv: {equiv_loss.item():.3} Hinge {entropy_reg.item():.3}")

    mu_loss /= len(data_loader)

    if mode == 'val':
        print(f"{mode.upper()} Epoch: {epoch}, Loss: {mu_loss:.3}")
        # Plot reconstruction
        save_image(recon[:16], f'{figures_dir}/recon_{str(epoch)}.png')
        errors.append(mu_loss)
        np.save(f'{MODEL_PATH}/errors_val.npy', errors)

        if (epoch % args.save_interval) == 0:
            save(model, model_file)

        if args.lr_scheduler:
            scheduler.step(loss)



if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch, train_loader, 'train')
        with torch.no_grad():
            train(epoch, val_loader, 'val')
