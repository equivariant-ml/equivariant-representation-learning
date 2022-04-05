import torch
from PIL import Image
from glob import glob
import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import pickle
from random import shuffle
import random

from torchvision import transforms

def convert_image(path):
    tmp = Image.open(path)
    tmp = tmp.resize((64,64), Image.ANTIALIAS)
    return np.asarray(tmp)[:,:,:3].transpose([2,0,1]) / 255.0

def convert_rotation(dic, name):
    r = R.from_euler('xyz', [0, dic[name]['elevation'], dic[name]['azimuth'] ], degrees=True)
    return r.as_matrix()

class ShapenetDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, mode, num_classes, clear=False):
        super().__init__()

        folders = sorted(glob(os.path.join(data_path, '*')))

        self.data = []

        self.train_data = []
        self.test_data = []

        train_pairs_path = data_path + f'/train_pairs_{num_classes}.pkl'
        test_pairs_path = data_path + f'/test_pairs_{num_classes}.pkl'

        if clear:
            print("Clearing old pairs")
            if os.path.exists(train_pairs_path):
                os.remove(train_pairs_path)
            if os.path.exists(test_pairs_path):
                os.remove(test_pairs_path)

        if mode == 'train':
            print("Train Pairs path:", train_pairs_path)
        else:
            print("Test Pairs path:", test_pairs_path)

        if not os.path.exists(train_pairs_path) and not os.path.exists(test_pairs_path):
            print("Building dataset...")
            for folder in tqdm(folders[:num_classes]):

                img_paths = sorted(glob(folder + '/*.png'))
                random.Random(4).shuffle(img_paths) # in-place operation

                split = int(0.7 * len(img_paths))

                train_img_paths = img_paths[0: split]
                test_img_paths = img_paths[(split + 1):]


                train_pairs = self.create_image_pairs(train_img_paths, folder)
                test_pairs = self.create_image_pairs(test_img_paths, folder)

                self.train_data += train_pairs
                self.test_data += test_pairs

            
            with open(train_pairs_path, 'wb') as f:
                pickle.dump(self.train_data, f)
            with open(test_pairs_path, 'wb') as f:
                pickle.dump(self.test_data, f)
            print("Built dataset!")
            
        if mode == 'train':
            with open(train_pairs_path, 'rb') as f:
                self.data = pickle.load(f)
        elif mode == 'test':
            with open(test_pairs_path, 'rb') as f:
                self.data = pickle.load(f)

        print("Num pairs: ", len(self.data))


    def create_image_pairs(self, img_paths, folder):
        num_images = len(img_paths)

        folder_data = []
        with open(folder + '/render_params.json') as f:
            render_params = json.load(f)

        for i in range(num_images - 1):
            for j in range(i+1, num_images):
                img1_path = img_paths[i]
                img2_path = img_paths[j]

                img_number1 = img1_path.split('/')[-1].split('.')[0]
                img_number2 = img2_path.split('/')[-1].split('.')[0]

                rot1 = convert_rotation(render_params, img_number1)
                rot2 = convert_rotation(render_params, img_number2)
                rot = rot2 @ rot1.T

                folder_data.append({
                    'img1_path': img1_path,
                    'img2_path': img2_path,
                    'rotation': rot
                })

        return folder_data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img1_path = self.data[idx]['img1_path']
        img2_path = self.data[idx]['img2_path']
        rotation = self.data[idx]['rotation']

        img1 = convert_image(img1_path)
        img2 = convert_image(img2_path)

        img1 = torch.from_numpy(img1).float()
        img2 = torch.from_numpy(img2).float()
        rotation = torch.from_numpy(rotation).float()

        return img1, img2, rotation, torch.Tensor([0])


if __name__ == '__main__':
    data = ShapenetDataset('data/shapenet/mugs/mugs-hq-train/mugs-hq-train')
    dloader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=True)
    
    img1,img2,rot = next(iter(dloader))

    print(img1.shape)
    
