import os
import h5py
import numpy as np
from tqdm import tqdm

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)

root = "/home/burak/workspace/seg/data/"

ALL_FILES = getDataFiles(os.path.join(root,'indoor3d_sem_seg_hdf5_data/all_files.txt'))
ALL_FILES = [os.path.join(root, f) for f in ALL_FILES]
room_filelist = [line.rstrip() for line in open(os.path.join(root,'indoor3d_sem_seg_hdf5_data/room_filelist.txt'))]

# Load ALL data
data_batch_list = []
label_batch_list = []
for h5_filename in tqdm(ALL_FILES, ncols=100, desc="Loading dataset into RAM"):
    data_batch, label_batch = loadDataFile(h5_filename)
    data_batch_list.append(data_batch)
    label_batch_list.append(label_batch)
data_batches = np.concatenate(data_batch_list, 0)
label_batches = np.concatenate(label_batch_list, 0)
print(data_batches.shape)
print(label_batches.shape)

# test_area = 'Area_'+str(FLAGS.test_area)
# train_idxs = []
# test_idxs = []
# for i,room_name in enumerate(room_filelist):
#     if test_area in room_name:
#         test_idxs.append(i)
#     else:
#         train_idxs.append(i)

# train_data = data_batches[train_idxs,...]
# train_label = label_batches[train_idxs]
# test_data = data_batches[test_idxs,...]
# test_label = label_batches[test_idxs]
# print(train_data.shape, train_label.shape)
# print(test_data.shape, test_label.shape)

