from multiprocessing.sharedctypes import Value
import torch
import numpy as np
import pickle
import scipy
import ipdb


class CustomTrajectory(torch.utils.data.Dataset):
    def __init__(self, T=100):
        """
        T = trajectory length
        """

        self.T = T
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

def CustomMergedTrajectory(data_dir, dataset, trajectory_length):
    if dataset == 'platonics':
        d1 = CustomDatasetTrajectory('tetra_black', trajectory_length, dataset, data_dir=data_dir)
        d2 = CustomDatasetTrajectory('cube_black', trajectory_length, dataset, data_dir=data_dir)
        d3 = CustomDatasetTrajectory('octa_black', trajectory_length, dataset, data_dir=data_dir)
    elif dataset == 'chairs':
        d1 = CustomDatasetTrajectory('shapenet_chair1', trajectory_length, dataset, data_dir=data_dir)
        d2 = CustomDatasetTrajectory('shapenet_chair2', trajectory_length, dataset, data_dir=data_dir)
        d3 = CustomDatasetTrajectory('shapenet_chair3', trajectory_length, dataset, data_dir=data_dir)
    return torch.utils.data.ConcatDataset([d1, d2, d3])

def CustomMerged(data_dir, dataset):
    d1 = CustomDataset('shapenet_chair1', dataset, data_dir=data_dir)
    d2 = CustomDataset('shapenet_chair2', dataset, data_dir=data_dir)
    d3 = CustomDataset('shapenet_chair3', dataset, data_dir=data_dir)
    return torch.utils.data.ConcatDataset([d1, d2, d3])

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, item, dataset, data_dir='data', logarithmic=False):

        self.classes = {'shapenet_chair1':0, 'shapenet_chair2':1, 'shapenet_chair3':2}
        assert item in list(self.classes.keys()), f"item must be one of {list(self.classes.keys())}"

        self.item = item
        self.logarithmic = logarithmic

        path = f'{data_dir}/shapenet_chair/{item}_data.pkl'
        with open(path, 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, idx):
        img1, img2, action = self.data[idx]

        img1 = torch.from_numpy(img1).float().permute(2, 0, 1) / 255.
        img2 = torch.from_numpy(img2).float().permute(2, 0 ,1) / 255.

        if self.logarithmic:
            action = scipy.linalg.logm(action)
        action = torch.from_numpy(action).float()


        return img1, img2, action, torch.Tensor([self.classes[self.item]]).long()

    def __len__(self):
        return len(self.data)


class CustomDatasetTrajectory(torch.utils.data.Dataset):
    def __init__(self, item, trajectory_length, dataset, data_dir='data', logarithmic=False):

        self.classes = {'shapenet_chair1':0, 'shapenet_chair2':1, 'shapenet_chair3':2}

        assert item in list(self.classes.keys()), f"item must be one of {list(self.classes.keys())}"

        self.item = item
        self.logarithmic = logarithmic

        path = f'{data_dir}/shapenet_chair/{item}_trajectory_{trajectory_length+1}.pkl'
        self.trajectory_length = trajectory_length

        print(f"Loading {path}")
        with open(path, 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, idx):
        img1, img2, action = self.data[idx]

        img1 = torch.from_numpy(img1).float().permute(2, 0, 1) / 255.
        img2 = torch.from_numpy(img2).float().permute(2, 0 ,1) / 255.

        if self.logarithmic:
            action = scipy.linalg.logm(action)
        action = torch.from_numpy(action).float()

        return img1, img2, action, torch.Tensor([self.classes[self.item]]).long()

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    pass
