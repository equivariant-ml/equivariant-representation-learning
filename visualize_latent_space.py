from multiprocessing.sharedctypes import Value
from tkinter import E
from comet_ml import Experiment

from sklearn.decomposition import PCA
import torch
import torch.utils.data
from torch import nn, optim, save, load
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchvision import transforms
import numpy as np
from scipy.spatial.transform import Rotation as R
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
from datasets.custom_dset import CustomMerged

from ENR.models_ENR import NeuralRenderer
from utils.utils import CometDummy, get_hinge_loss
from matplotlib import cm
import ipdb


parser = argparse.ArgumentParser()
parser.add_argument('--save-folder', type=str, default='checkpoints', help='Path to saved model')
args_eval = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.manual_seed(42)
torch.cuda.empty_cache()

model_file = os.path.join(args_eval.save_folder, 'model.pt')
meta_file = os.path.join(args_eval.save_folder, 'metadata.pkl')
args = pickle.load(open(meta_file, 'rb'))['args']


if args.dataset == 'color-shift':
    dset = EquivDataset('data/colorshift_data/')
    action_dim = 3
    group_dim = 3
elif args.dataset == 'sprites':
    dset = EquivDataset('data/sprites_data/', greyscale = True)
    action_dim = 3
    group_dim = 3
elif args.dataset == 'multi-sprites':
    dset = EquivDataset('data/multisprites_data/')
    action_dim = 6
    group_dim = 6
elif args.dataset == 'platonics' or args.dataset in ['mugs', 'chairs'] or args.dataset in ['custom_chair'] or args.dataset in ['custom_shapenet']:
    dset = CustomMerged(data_dir=args.data_dir, dataset=args.dataset)
    action_dim = 3
    group_dim = 9
    if args.method == 'naive':
        action_dim = 3
        group_dim = 3
elif args.dataset == 'room_combined':
    #dset = EquivImgDataset('./data/combine/', combined=True)
    dset = EquivDataset('./data/gibson_global_frame/')
    action_dim = 3
    group_dim = 4
elif args.dataset == 'room_combined_local':
    #dset = EquivImgDataset('./data/combine/', combined=True)
    dset = EquivDataset('./data/gibson_local_frame/')
    action_dim = 3
    group_dim = 4
else:
    print("Invalid dataset")

train_loader = torch.utils.data.DataLoader(dset,
                                           batch_size=300, shuffle = True)

device = 'cpu'
model = load(model_file).to(device)
model.eval()


img, img_next, action, classes = next(iter(train_loader))
img_shape = img.shape[1:]


z, z_pre = model.encode_pose(img.to(device))
extra = z[:, group_dim: ].detach().cpu().numpy()
pose = z[:, :group_dim].detach().cpu().numpy()
classes = classes.flatten().detach().cpu().numpy()
classes_norm = classes / classes.max()


# for i in range(10):
#     plt.imshow(np.transpose(img[i].detach().cpu().numpy(), axes=[1,2,0])*255)
#     plt.show()
#     plt.imshow(np.transpose(img_next[i].detach().cpu().numpy(), axes=[1,2,0])*255)
#     plt.show()
#     print(extra[i])


def mymap(x):
    if x == 1:
        return '#F97306'
    if x == 2:
        return '#069AF3'
    if x == 3 or x==0:
        return '#C1F80A'
#colors = [ cm.Spectral(x) for x in classes_norm ]
colors = [mymap(x) for x in classes]
#print(colors)

fig = plt.figure(figsize=plt.figaspect(1))
ax = fig.add_subplot(projection='3d')

if (args.dataset == 'platonics' or args.dataset in ['mugs', 'chairs'] or args.dataset in ['custom_chair'] or args.dataset in ['custom_shapenet']) and args.method != 'naive':
    rot = R.from_matrix(pose.reshape((-1,3,3)))
    angles = rot.as_euler('XYZ', degrees=False)
    ax.scatter(np.cos(angles[:, 0]), np.sin(angles[:, 0]), extra[:,1] + extra[:,0], c=colors)
    ax.view_init(elev=32., azim=50.)  #43 67  #27 63
    ax.grid(False)
    plt.tight_layout()
    plt.savefig('latent_rot.png')
    plt.show()

# emb = PCA(n_components=3)
# extra = emb.fit_transform(z.detach().cpu().numpy())
#ax.scatter(pose[:, 0], pose[:, 1], extra[:,0], c=colors)
#ax.scatter(extra[:, 0], extra[:, 1], extra[:, 2], c=colors)

ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], c=colors)

if args.method == 'naive':
    ax.scatter(pose[:, 0], pose[:, 1], pose[:,2], c=colors)
else:
    ax.scatter(pose[:, 0], pose[:, 1], extra[:,0], c=colors)
#ax.scatter(extra[:, 0], extra[:, 1], extra[:, 1], c=colors)
ax.view_init(elev=32., azim=50.)  #43 67  #27 63
ax.grid(False)
plt.tight_layout()
# plt.savefig('latent_space.png')
plt.show()

room = 2
if args.dataset == 'room_combined':
    fig = plt.figure()
    pose = pose[classes == room]
    print(pose)
    plt.scatter(pose[:,0], pose[:,1], color=mymap(room), s=100, alpha=.7)
    ax  = plt.gca()
    ax.set_aspect('equal')
    plt.axis('off')
    plt.savefig('room_map.png')
    plt.show()
