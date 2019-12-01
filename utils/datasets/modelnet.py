#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
"""


import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset


def download(DATA_DIR):
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(DATA_DIR, partition):
    download(DATA_DIR)
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, data_folder, split, path_prefix=None, augmentation=False, num_points=1024, channels_first=False):
        if path_prefix:
            data_folder = os.path.join(path_prefix, data_folder)
        self.data, self.label = load_data(data_folder, partition=split)
        self.num_points = num_points
        self.augmentation = augmentation        
        self.channels_first = channels_first        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.augmentation:
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)

        if self.channels_first:
            pointcloud = pointcloud.T
        return pointcloud, label[0]

    def __len__(self):
        return len(self.data)

def get_sets(data_folder, path_prefix=None, training_augmentation=True):
    """Return hooks to ModelNet40 dataset train, validation and tests sets.
    """

    train_set = ModelNet40(data_folder, 'train', path_prefix, num_points=1024, augmentation=training_augmentation)
    valid_set = ModelNet40(data_folder, 'test', path_prefix, num_points=1024)
    test_set = ModelNet40(data_folder, 'test', path_prefix, num_points=1024)

    return train_set, valid_set, test_set

if __name__ == '__main__':
    prfx = ""
    train = ModelNet40("data/modelnet", 'train', num_points=1024, augmentation=True, path_prefix=prfx)
    test = ModelNet40("data/modelnet", 'test', num_points=1024, path_prefix=prfx)
    for i, (data, label) in enumerate(train):
        print("\r%d. %s, %s"%(i, data.shape, label.shape), end="", flush=True)

    print()