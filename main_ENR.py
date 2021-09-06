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

from ENR.models_ENR import NeuralRenderer
from nn_utils import entropy_loss
from utils import make_dir

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
parser.add_argument('--dataset', default='platonics', type=str, help="Dataset")
parser.add_argument('--batch-size', type=int, default=64, help="Batch size")

# Model arguments
parser.add_argument('--model', default='resnet', type=str, help="Model to use")
parser.add_argument('--model-name', required=True, type=str, help="Name of model")
parser.add_argument('--checkpoints-dir', default='checkpoints', type=str)

# Optimization
parser.add_argument('--lr-scheduler', action='store_true', default=False, help="Use a lr scheduler")
parser.add_argument('--lr', default=1e-3, type=float)

parser.add_argument('--data-dir', type=str, default='data')

args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Save paths
MODEL_PATH = os.path.join(args.checkpoints_dir, args.model_name)

figures_dir = os.path.join(MODEL_PATH, 'figures')
model_file = os.path.join(MODEL_PATH, 'model.pt')
meta_file = os.path.join(MODEL_PATH, 'metadata.pkl')
log_file = os.path.join(MODEL_PATH, 'log.txt')

make_dir(MODEL_PATH)
make_dir(figures_dir)

pickle.dump({'args': args}, open(meta_file, 'wb'))

batch_size = args.batch_size


# Allocate dataset
if args.dataset == 'platonics':
    dset = PlatonicMerged(30000, big=True, data_dir=args.data_dir)
else:
    print("Invalid dataset")


train_data, valid_data = torch.utils.data.random_split(dset, [len(dset) - int(len(dset)/10), int(len(dset)/10)])
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(valid_data,
                                           batch_size=batch_size, shuffle=True, drop_last=True)

# Sample data
img, _, _, _ = next(iter(train_loader))
img_shape = img.shape[1:]


# Create Model
if args.model == 'cnn':
    model = CNN(img_shape[0]).to(device)
elif args.model == 'resnet':
    model = NeuralRenderer([3, 64, 64], [64, 64, 128, 128, 128, 128, 256, 256, 128, 128, 128],
                            [1, 1, 2, 1, 2, 1, 2, 1, -2, 1, 1], [32, 32, 128, 128, 128, 64, 64, 64],
                            [1, 1, 2, 1, 1, -2, 1, 1], [256, 512, 1024],  [512, 256, 256]).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)

if args.lr_scheduler:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20)

errors_train = []
def train(epoch, data_loader, mode='train'):

    mu_loss = 0

    global_step = epoch * len(data_loader)

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

        equiv_recon, recon = model(img, action)

        loss = entropy_loss(equiv_recon, img_next)

        with torch.no_grad():
            recon_next = model.decode(model.encode(img_next))
            val_loss = 0.5 * (entropy_loss(recon, img) + entropy_loss(recon_next, img_next))
            mu_loss += val_loss.item()


        if mode == 'train':
            loss.backward(retain_graph=True)
            optimizer.step()

        if batch_idx % args.log_interval == 0 and mode == 'train':
            print(f"{mode.upper()} Epoch: {epoch}, Batch: {batch_idx} / {len(train_loader)} \
                        Loss: {loss.item():.3}")


    mu_loss /= len(data_loader)

    if mode == 'val':
        print(f"{mode.upper()} Epoch: {epoch}, Loss: {mu_loss:.3}")
        # Plot reconstruction
        save_image(recon[:16], f'{figures_dir}/recon_{str(epoch)}.png')
        errors_train.append(mu_loss)
        np.save(f'{MODEL_PATH}/errors_val.npy', np.array(errors_train))

        if (epoch % args.save_interval) == 0:
            save(model, model_file)


        if args.lr_scheduler and mode == 'val':
            scheduler.step(loss)


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch, train_loader, 'train')
        train(epoch, val_loader, 'val')
