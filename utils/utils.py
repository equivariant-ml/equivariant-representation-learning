import torch
import torch.nn.functional as F
import ipdb
from datasets.platonic_dset import PlatonicMerged
from datasets.equiv_dset import *
from datasets.custom_dset import CustomMerged
from datasets.custom_dset import CustomMergedTrajectory
import ipdb


class CometDummy():
    def __init__(self):
        super()

    def log_metric(self, *args, **kwargs):
        pass

    def log_parameters(self, *args, **kwargs):
        pass

    def log_figure(self, *args, **kwargs):
        pass


## Loss functions

def get_hinge_loss(z_encoded, z_next_encoded, action_dim, model, method, decoder):
    # ipdb.set_trace()
    """
    hinge loss:
    in the case of MDP: push everything apart
    in everything else: put classes close, push random stuff apart
    """

    device = z_encoded.device
    batch_size = z_encoded.shape[0]
    if model == 'mdp' or model == 'mdp_resnet' or method == 'naive':
        ce_loss = torch.Tensor([0]).to(device)
        z = z_next_encoded
    else:
        ce_loss = F.mse_loss(z_encoded[:, action_dim:], z_next_encoded[:, action_dim:])
        z = z_next_encoded[:, action_dim:]

    z_random = z[torch.randperm(batch_size)]
    distance = torch.norm(z - z_random, p=2, dim=-1)
    hinge = torch.max(torch.zeros_like(distance).to(device), torch.ones_like(distance).to(device) - distance).mean()


    if method == 'lie_right' or (method == 'lie' and decoder):
        hinge = torch.Tensor([0]).to(device)

    return hinge + ce_loss


def load_dataset(dataset, data_dir, model, method):
    # data_dir = args_eval.data_dir
    # method = args.data_dir

    if dataset == 'color-shift':
        ACTION_TYPE='translate'
        dset = EquivDataset('data/colorshift_data/')
        action_dim = 3
        group_dim = 3
        dset_10 = EquivDataset(f'{data_dir}/colorshift_data/', length_trajectory=10)
        dset_20 = EquivDataset(f'{data_dir}/colorshift_data/', length_trajectory=20)
    elif dataset == 'sprites':
        ACTION_TYPE='translate'
        dset = EquivDataset('data/sprites_data/', greyscale = True)
        action_dim = 3
        group_dim = 3
        dset_10 = EquivDataset(f'{data_dir}/sprites_data/', greyscale=True, length_trajectory=10)
        dset_20 = EquivDataset(f'{data_dir}/sprites_data/', greyscale=True, length_trajectory=20)
    elif dataset == 'multi-sprites':
        ACTION_TYPE='translate'
        dset = EquivDataset('data/multisprites_data/')
        action_dim = 6
        group_dim = 6
        dset_10 = EquivDataset(f'{data_dir}/multisprites_data/', length_trajectory=10)
        dset_20 = EquivDataset(f'{data_dir}/multisprites_data/', length_trajectory=20)
    elif dataset == 'platonics' or dataset in ['mugs', 'chairs'] or dataset in ['custom_chair'] or dataset in ['custom_shapenet']:
        ACTION_TYPE='rotate'
        dset = CustomMerged(data_dir=data_dir, dataset=dataset)
        action_dim = 3
        group_dim = 9
        if method == 'naive':
            action_dim = 3
            group_dim = 3
        dset_10 = CustomMergedTrajectory(data_dir=data_dir, dataset=dataset, trajectory_length=10)
        dset_20 = CustomMergedTrajectory(data_dir=data_dir, dataset=dataset, trajectory_length=20)
    elif dataset == 'room_combined':
        ACTION_TYPE = 'isometries_2d'
        dset = EquivDataset('./data/gibson_global_frame/')
        action_dim = 3
        group_dim = 4
        dset_10 = EquivDataset(f'{data_dir}/gibson_global_frame/', length_trajectory=10)
        dset_20 = EquivDataset(f'{data_dir}/gibson_global_frame/', length_trajectory=20)
    elif dataset == 'room_combined_local':
        ACTION_TYPE = 'isometries_2d_local'
        dset = EquivDataset('./data/gibson_local_frame/')
        action_dim = 3
        group_dim = 4
        dset_10 = EquivDataset(f'{data_dir}/gibson_local_frame/', length_trajectory=10)
        dset_20 = EquivDataset(f'{data_dir}/gibson_local_frame/', length_trajectory=20)
    else:
        print("Invalid dataset")
        raise ValueError

    if model == 'mdp' or model == 'mdp_resnet':
        group_dim = 0

    return dset, dset_10, dset_20, action_dim, group_dim, ACTION_TYPE