import os
import sys
import glob

from tqdm import tqdm
import numpy as np
import torch.utils.data as data

# Add parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import augmentations


SPLITS = {
    1: {
        "train": ["Area_2", "Area_3", "Area_4", "Area_5", "Area_6"],
        "test": ["Area_1"],
        "val": ["Area_1"],
    },
    2: {
        "train": ["Area_1", "Area_3", "Area_4", "Area_5", "Area_6"],
        "test": ["Area_2"],
        "val": ["Area_2"],
    },
    3: {
        "train": ["Area_1", "Area_2", "Area_4", "Area_5", "Area_6"],
        "test": ["Area_3"],
        "val": ["Area_3"],
    },
    4: {
        "train": ["Area_1", "Area_2", "Area_3", "Area_5", "Area_6"],
        "test": ["Area_4"],
        "val": ["Area_4"],
    },
    5: {
        "train": ["Area_1", "Area_2", "Area_3", "Area_4", "Area_6"],
        "test": ["Area_5"],
        "val": ["Area_5"],
    },
    6: {
        "train": ["Area_1", "Area_2", "Area_3", "Area_4", "Area_5"],
        "test": ["Area_6"],
        "val": ["Area_6"],
    },
}


class S3DISDataset(data.Dataset):
    def __init__(self, root="data/Stanford3d_batch_version", split="test", augmentation=False):
        self.root = root
        self.cls_list=['clutter', 'ceiling', 'floor', 'wall', 'beam', 'column',
                    'door', 'window', 'table', 'chair', 'sofa', 'bookcase', 'board']

        if isinstance(split, str):
            split_areas = {
                "train": ["Area_1", "Area_2", "Area_3", "Area_4", "Area_6"],
                "test": ["Area_5"],
                "val": ["Area_5"],
            }
            areas = split_areas[split]
        elif isinstance(split, list):
            areas = split
        else:
            raise ValueError("Unsupported split type.")

        self.augmentation = augmentation

        self.room_list = self.create_room_list(areas)
        self.cell_list, self.room_cloud_files = self.create_cell_list()

        self.num_labels = 13    # Including negative class
        self.labelweights = np.ones(self.num_labels, dtype=np.float32)


    def create_room_list(self, area_list):
        room_list = []
        for area in area_list:
            area_pattern = os.path.join(self.root, area, "[!.]*")
            room_list += [room for room in glob.glob(area_pattern)]

        return room_list


    def create_cell_list(self):
        cells, room_clouds = [], []
        for room in tqdm(self.room_list, desc="Reading dataset", ncols=100):
            room_cell_path = os.path.join(room, 'cell')
            object_list = sorted(os.listdir(room_cell_path))
            for object_name in object_list:
                object_path = os.path.join(room_cell_path, object_name)
                cells += [os.path.join(object_path, c) for c in os.listdir(object_path)]
                room_clouds += [os.path.join(room, "whole_room_point.npy") for c in os.listdir(object_path)]

        return cells, room_clouds

    def augment_data(self, data, label):
        
        batch_data = data[np.newaxis, :, :3]

        # batch_data, label, _ = augmentations.shuffle_data(batch_data, label)
        batch_data = augmentations.rotate_point_cloud(batch_data, "z")
        # batch_data = augmentations.shift_point_cloud(batch_data)
        batch_data = augmentations.random_scale_point_cloud(batch_data)
        # batch_data = augmentations.jitter_point_cloud(batch_data)

        data = np.hstack((batch_data[0], data[:, 3:6]))

        return data, label


    def __getitem__(self, i):
        cell_indices = np.load(self.cell_list[i])
        room_cloud = np.load(self.room_cloud_files[i])
        data = room_cloud[cell_indices]

        cell_cloud = data[:, :6]
        cell_cloud[:, 3:6] = cell_cloud[:, 3:6] / 255 - 127.5
        label = data[:, 6].astype(np.int64)

        if self.augmentation:
            cell_cloud, label = self.augment_data(cell_cloud, label)

        # Make it channels first
        cell_cloud = np.swapaxes(cell_cloud, 0, 1).astype(np.float32)

        return cell_cloud, label

    def __len__(self):
        return len(self.cell_list)

def get_sets(data_folder, crossval_id=None, training_augmentation=True):
    """Return hooks to S3DIS dataset train, validation and tests sets.
    """

    train_set = S3DISDataset(data_folder, split=SPLITS[crossval_id]['train'], augmentation=training_augmentation)
    valid_set = S3DISDataset(data_folder, split=SPLITS[crossval_id]['val'])
    test_set = S3DISDataset(data_folder, split=SPLITS[crossval_id]['test'])

    return train_set, valid_set, test_set

def test():
    # from svstools import visualization as vis
    datafolder='/data1/jiajing/dataset/S3DIS_cell_version/Stanford3dDataset_v1.2_Aligned_Version'
    t, _, _ = get_sets(datafolder, crossval_id=5, training_augmentation=False)

    for i in range(10,20):
        X, y = t[i]

        pcd = vis.paint_colormap(X[-3:].T, y)
        vis.show_pointcloud(pcd)

    return

if __name__=='__main__':
    test()
