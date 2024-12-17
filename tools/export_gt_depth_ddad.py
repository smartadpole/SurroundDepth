import os
from dgp.datasets import SynchronizedSceneDataset
from tqdm import tqdm
import copy
import numpy as np
import pickle
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Export ground truth depth for DDAD dataset")
    parser.add_argument("--root_path", type=str, default='../data/ddad/raw_data', help="Root path to the raw data")
    parser.add_argument("--split", type=str, default='train', help="Dataset split to use (e.g., train, val, test)")
    parser.add_argument("--index_path", type=str, default='../datasets/ddad/index.pkl', help="Path to the index file")

    return parser.parse_args()


# convert lidar to depth
def load_dataset(root_path, split):
    dataset = SynchronizedSceneDataset(os.path.join(root_path, 'ddad.json'),
                                       datum_names=('CAMERA_01', 'CAMERA_05', 'CAMERA_06', 'CAMERA_07', 'CAMERA_08', 'CAMERA_09', 'lidar'),
                                       generate_depth_from_datum='lidar',
                                       split=split)
    return dataset

def load_index_info(index_path):
    with open(index_path, 'rb') as f:
        index_info = pickle.load(f)
    return index_info

def save_depth_data(dataset, index_info, root_path, camera_names):
    to_save = {}
    print(len(dataset))
    for i in range(200):
        for camera_name in camera_names:
            os.makedirs(os.path.join(root_path, '{:06d}'.format(i), 'depth', camera_name), exist_ok=True)

    count = 0
    for data in tqdm(dataset):
        count += 1
        for i in range(6):
            m = data[0][i]
            t = str(m['timestamp'])
            save_temp = copy.deepcopy(m)
            if t not in to_save.keys():
                to_save[t] = copy.deepcopy(index_info[t])

            scene_id = index_info[t]['scene_name']
            save_path = os.path.join(root_path, scene_id, 'depth', m['datum_name'], t + '.npy')
            np.save(save_path, m['depth'])

def main():
    args = get_args()
    root_path = args.root_path
    split = args.split
    index_path = args.index_path
    dataset = load_dataset(root_path, split)
    root_path = os.path.join(root_path, 'depth')
    camera_names = ['CAMERA_01', 'CAMERA_05', 'CAMERA_06', 'CAMERA_07', 'CAMERA_08', 'CAMERA_09']
    index_info = load_index_info(index_path)
    save_depth_data(dataset, index_info, root_path, camera_names)

if __name__ == "__main__":
    main()