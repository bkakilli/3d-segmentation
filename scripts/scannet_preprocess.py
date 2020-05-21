import os
import glob
import json

import numpy as np
import open3d as o3d
from tqdm import tqdm

np.set_printoptions(suppress=True)
np.random.seed(0)

LABEL_DICT = {
    "wall" : 1,
    "floor" : 2,
    "cabinet" : 3,
    "bed" : 4,
    "chair" : 5,
    "sofa" : 6,
    "table" : 7,
    "door" : 8,
    "window" : 9,
    "bookshelf" : 10,
    "picture" : 11,
    "counter" : 12,
    "desk" : 14,
    "curtain" : 16,
    "refridgerator" : 24,
    "shower curtain" : 28,
    "toilet" : 33,
    "sink" : 34,
    "bathtub" : 36,
    "otherfurniture" : 39,
}

def load_capture(capture_name, capture_path, split):

    # Load XYZ and Color
    ply_path = os.path.join(capture_path, capture_name+'_vh_clean_2.ply')
    pcd = o3d.io.read_point_cloud(ply_path)

    points = np.asarray(pcd.points).astype(np.float32)
    color = np.asarray(pcd.colors).astype(np.float32)

    labels = np.ones((len(points), 1), dtype=np.float32) * (-1)
    
    if split is "test":
        return points, color, labels

    # Read labels    
    agg_file_path = os.path.join(capture_path, capture_name+'.aggregation.json')
    with open(agg_file_path) as f:
        aggregation = json.loads(f.read())

    
    element_list=[]
    cls_list=[]
    other_id = len(LABEL_DICT)
    for seg_group in aggregation['segGroups']:
        class_name = seg_group['label']
        label_id = LABEL_DICT[class_name] if class_name in LABEL_DICT else other_id
        
        segments = seg_group['segments']
        element_list += segments
        cls_list += len(segments)*[label_id]
    
    segment_file_path = os.path.join(capture_path, capture_name+'_vh_clean_2.0.010000.segs.json')
    with open(segment_file_path) as f:
        segmentation = json.loads(f.read())

    all_points_list = np.asarray(segmentation['segIndices'])

    for i, point_i in enumerate(element_list):
        labels[all_points_list==point_i] = cls_list[i]

    return points, color, labels

def create_dataset(scannet_root, save_path, num_points=8192):

    # List all captures in all scenes
    loaded = {}
    for split in ["train", "val", "test"]:
        split_list_file = os.path.join(scannet_root, "Tasks/Benchmark/scannetv2_%s.txt"%split)
        split_list = np.loadtxt(split_list_file, dtype=str).tolist()
        
        sub_folder = "_test" if split is "test" else ""
        split_list_path = [os.path.join(scannet_root, "DATA/scans%s"%sub_folder, c) for c in split_list]

        tqdm_iterator = tqdm(zip(split_list, split_list_path), desc="Loading %s"%split, ncols=100, total=len(split_list))
        for capture_name, capture_path in tqdm_iterator:
            xyz, rgb, label = load_capture(capture_name, capture_path, split)
            concatenated = np.hstack((xyz, rgb, label))
            loaded[split+"/"+capture_name] = concatenated

    save_path = os.path.join(save_path, "preloaded.npz")
    print("Saving preloaded dataset into %s" % save_path)
    np.savez_compressed(save_path, **loaded)
    

def take_cell_sample(capture_data, sample_at=None, num_points=8192, dims=None, method="random", min_N=256):

    if dims is None:
        dims = [1.0, 1.0, 99.0]
    dims = np.asarray(dims)
    
    if method is "random":

        def random_sample():
            min_coords = capture_data[:, :3].min(axis=0)
            max_coords = capture_data[:, :3].max(axis=0)
            span = np.array(max_coords - min_coords)

            # Origin
            o = min_coords + np.random.rand(3)*span - dims/2

            mask = np.logical_and.reduce((
                capture_data[:, 0] > o[0],
                capture_data[:, 1] > o[1],
                capture_data[:, 0] <= o[0]+dims[0],
                capture_data[:, 1] <= o[1]+dims[1],
            ))

            return capture_data[mask]

        # Try sampling until we have sufficient points
        sampled = random_sample()
        while len(sampled) < min_N:
            sampled = random_sample()

        # Sample additional points (copies) if total is less then desired
        if len(sampled) < num_points:
            additional_samples = sampled[np.random.choice(len(sampled), num_points-len(sampled), replace=True)]
            sampled = np.vstack((sampled, additional_samples))
        else:
            sampled = np.random.permutation(sampled)[:num_points]
        
        return sampled

def load_test(load_path):

    load_path = os.path.join(load_path, "preloaded.npz")

    loaded = np.load(load_path)
    files = loaded.files
    train_files = [f for f in files if "train" in f]

    train_batch = [loaded[f] for f in train_files[:16]]

    sample = take_cell_sample(train_batch[0])
    
    return


def main():
    scannet_root = '/data1/datasets/ScanNet'
    save_path = "/data1/datasets/scannet_preprocessed"

    # create_dataset(scannet_root, save_path, num_points=8192)
    load_test(save_path)


if __name__=='__main__':
    main()
