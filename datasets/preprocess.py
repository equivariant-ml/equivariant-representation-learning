import numpy as np
from PIL import Image
import os
import glob
import ipdb

from platonic_dset import PlatonicDataset
from custom_dset import CustomDataset
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import pickle
import ipdb
import matplotlib.pyplot as plt


def extract_rotation_matrix(path):
    # ipdb.set_trace()
    
    quat_str = path.split('_')[-1].split('.png')[0]
    quats = [float(x) for x in quat_str.split('&')]

    r = R.from_quat(quats)
    return r.as_matrix()

def get_action(path1, path2):
    pose1 = extract_rotation_matrix(path1)
    pose2 = extract_rotation_matrix(path2)

    # LEFT ACTION
    action = pose2 @ np.linalg.inv(pose1)
    return action


def generate_dataset(save_path, files):
    num_pairs = len(files)
    data = []
    for i in tqdm(range(0, num_pairs, 2)):
        img1_path = files[i]
        img2_path = files[i+1]

        action = get_action(img1_path, img2_path)
        img1 = np.asarray(Image.open(img1_path).resize((64,64)))[:,:,:3]
        img2 = np.asarray(Image.open(img2_path).resize((64,64)))[:,:,:3]

        # plt.figure()
        # plt.imshow(img1)
        # plt.show()
        #ipdb.set_trace()

        data.append((img1, img2, action))

    print("FInal path", img1_path)
    print("FInal path", img2_path)

    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

    print("Saved at:", save_path)



if __name__ == '__main__':
    data_root_dir = 'data/blender_render'
    #obj_type = 'shapenet_chair3'
    #obj_type = 'octa_black'
    #objs = ['octa_black', 'tetra_black']#, 'cube_black']
    objs = ['cube_black']
    for obj_type in objs:
        save_path = os.path.join(data_root_dir, obj_type + '_data.pkl')

        data_path = os.path.join(data_root_dir, obj_type)

        files = sorted(glob.glob(data_path + '/*'), key=lambda x : int(x.split('/')[-1].split('_')[3]))
        print("num files", len(files))

        generate_dataset(save_path, files)
        print(f"Done generating {obj_type}")

    # dset = CustomDataset('chair')
    # img1, img2, action, clazz = dset[0]





    # platonics = PlatonicDataset('cube', N=10000)
    # img1, img2, action, clazz = platonics[0]
    # print("ACTION")
    # print(action)