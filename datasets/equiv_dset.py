import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from PIL import Image
import random
import ipdb


class EquivDataset(torch.utils.data.Dataset):
    def __init__(self, PATH, greyscale = False, length_trajectory=1):
        if length_trajectory==1:
            self.data = np.load(PATH + 'equiv_data.npy',  mmap_mode = 'r+')
            self.lbls = np.load(PATH + 'equiv_lbls.npy',  mmap_mode = 'r+')
            self.classes = np.load(PATH + 'equiv_classes.npy',  mmap_mode = 'r+')
        else:
            self.data = np.load(PATH + f'equiv_data_{length_trajectory}.npy',  mmap_mode = 'r+')
            self.lbls = np.load(PATH + f'equiv_lbls_{length_trajectory}.npy',  mmap_mode = 'r+')
            self.classes = np.load(PATH + f'equiv_classes_{length_trajectory}.npy',  mmap_mode = 'r+')
        self.greyscale = greyscale

    def __getitem__(self, index):
        if self.greyscale:
            return torch.FloatTensor(self.data[index, 0]).unsqueeze(0), torch.FloatTensor(self.data[index, 1]).unsqueeze(0), torch.FloatTensor(self.lbls[index]), torch.FloatTensor((self.classes[index],))
        else:
            return torch.FloatTensor(self.data[index, 0]), torch.FloatTensor(self.data[index, 1]), torch.FloatTensor(self.lbls[index]), torch.FloatTensor((self.classes[index],))



    def __len__(self):
        return len(self.data)


class EquivImgDataset(torch.utils.data.Dataset):
    def __init__(self, PATH, grayscale=False, combined=False):
        self.lbls = np.load(PATH + 'equiv_lbls.npy',  mmap_mode = 'c')
        self.PATH = PATH
        self.grayscale = grayscale
        self.combined = combined
        if combined:
            self.classes = np.load(PATH + 'classes.npy')

    def __getitem__(self, i):
        tmp = Image.open(self.PATH + 'img' + str(i) + '.png')
        tmp.load()
        tmp_next = Image.open(self.PATH + 'imgnext' + str(i) + '.png')
        tmp_next.load()

        if self.grayscale:
            img = np.expand_dims(np.asarray(tmp, dtype="float32"), axis=0 ) / 255
            img_next = np.expand_dims(np.asarray(tmp_next, dtype="float32"), axis=0 ) / 255

        else:
            img = np.transpose(np.asarray(tmp, dtype="float32"), (2,0,1))[:3,:,:] /255  #Images are assumed .png in RGBA
            img_next = np.transpose(np.asarray(tmp_next, dtype="float32"), (2,0,1))[:3,:,:] /255

        #print(self.classes.shape)
        if self.combined:
            return torch.FloatTensor(img), torch.FloatTensor(img_next), torch.FloatTensor(self.lbls[i]), torch.FloatTensor((self.classes[i, 1], ))
        else:
            return torch.FloatTensor(img), torch.FloatTensor(img_next), torch.FloatTensor(self.lbls[i]), 0


    def __len__(self):
            return len(self.lbls)
