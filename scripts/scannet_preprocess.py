import os
import glob
import json

import numpy as np
import open3d as o3d
from tqdm import tqdm

np.set_printoptions(suppress=True)
np.random.seed(0)

LABEL_DICT = {
    "unannotated":      0,
    "wall" :            1, # 1,
    "floor" :           2, # 2,
    "cabinet" :         3, # 3,
    "bed" :             4, # 4,
    "chair" :           5, # 5,
    "sofa" :            6, # 6,
    "table" :           7, # 7,
    "door" :            8, # 8,
    "window" :          9, # 9,
    "bookshelf" :       10, # 10,
    "picture" :         11, # 11,
    "counter" :         12, # 12,
    "desk" :            13, # 14,
    "curtain" :         14, # 16,
    "refridgerator" :   15, # 24,
    "shower curtain":   16, # 28,
    "toilet" :          17, # 33,
    "sink" :            18, # 34,
    "bathtub" :         19, # 36,
    "otherfurniture" :  20, # 39
}


def load_capture(capture_name, capture_path, split, raw2nyu40_map):

    # Load XYZ and Color
    ply_path = os.path.join(capture_path, capture_name+'_vh_clean_2.ply')
    pcd = o3d.io.read_point_cloud(ply_path)

    points = np.asarray(pcd.points).astype(np.float32)
    color = np.asarray(pcd.colors).astype(np.float32)

    labels = np.zeros((len(points), 1), dtype=np.float32)
    instances = np.zeros((len(points), 1), dtype=np.float32)-1
    
    if split is "test":
        return points, color, labels, instances

    # Read labels
    labels -= 1 # Make all of them -1

    agg_file_path = os.path.join(capture_path, capture_name+'.aggregation.json')
    with open(agg_file_path) as f:
        aggregation = json.loads(f.read())

    
    segments_list = []
    label_list = []
    instance_list = []
    for instance, seg_group in enumerate(aggregation['segGroups']):
        raw_class_name = seg_group['label']
        class_name = raw2nyu40_map[raw_class_name]
        label_id = LABEL_DICT[class_name] if class_name in LABEL_DICT else LABEL_DICT["unannotated"]
        
        segments = seg_group['segments']
        segments_list += segments
        label_list += len(segments)*[label_id]
        instance_list += len(segments)*[seg_group["id"]]

        if instance != seg_group["id"] or instance != seg_group["objectId"]:
            raise ValueError("Not equal")
    
    segment_file_path = os.path.join(capture_path, capture_name+'_vh_clean_2.0.010000.segs.json')
    with open(segment_file_path) as f:
        segmentation = json.loads(f.read())

    all_points_list = np.asarray(segmentation['segIndices'])

    for i, point_i in enumerate(segments_list):
        segment_indices = all_points_list==point_i
        labels[segment_indices] = label_list[i]
        instances[segment_indices] = instance_list[i]

    # Select only valid items
    mask = labels.reshape(-1) != -1

    return points[mask], color[mask], labels[mask], instances[mask]

def create_dataset(scannet_root, save_path):

    # Get label name mapping
    tsv_file = os.path.join(scannet_root, 'DATA/scannetv2-labels.combined.tsv')
    table = np.loadtxt(tsv_file, dtype=str, delimiter='\t', usecols=[1,7], skiprows=1)
    raw2nyu40_map = {raw: nyu40 for raw, nyu40 in table}

    # List all captures in all scenes
    loaded = {}
    for split in ["train", "val", "test"]:
        split_list_file = os.path.join(scannet_root, "Tasks/Benchmark/scannetv2_%s.txt"%split)
        split_list = np.loadtxt(split_list_file, dtype=str)
        
        sub_folder = "_test" if split is "test" else ""
        split_list_path = [os.path.join(scannet_root, "DATA/scans%s"%sub_folder, c) for c in split_list]

        loaded[split+"/count"] = np.zeros(len(LABEL_DICT), dtype=np.int64)
        loaded[split+"/cells"] = np.zeros(len(LABEL_DICT), dtype=np.int64)

        tqdm_iterator = tqdm(zip(split_list, split_list_path), desc="Loading %s"%split, ncols=100, total=len(split_list))
        for capture_name, capture_path in tqdm_iterator:
            xyz, rgb, label, instance = load_capture(capture_name, capture_path, split, raw2nyu40_map)
            concatenated = np.hstack((xyz, rgb, label, instance))

            loaded[split+"/captures/"+capture_name] = concatenated
            loaded[split+"/count"] += np.bincount(label.reshape(-1).astype(int), minlength=len(LABEL_DICT))

            # Calculate cells for training and validation tests
            if (instance+1).sum() > 0:
                cells = objectwise_cell_sampling(concatenated)
                for i, cell in enumerate(cells):
                    cell_id = "%03d" % i
                    loaded[split+"/cells/"+capture_name+"_cell_"+cell_id] = cell

    save_path = os.path.join(save_path, "preloaded_512.npz")
    print("Saving preloaded dataset into %s" % save_path)
    np.savez(save_path, **loaded)
    

def take_random_cell_sample(capture_data, sample_at=None, num_points=8192, dims=None, method="random", min_N=256):

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
            duplicate_indices = sampled[np.random.choice(len(sampled), num_points-len(sampled), replace=True)]
            sampled = np.vstack((sampled, duplicate_indices))
        else:
            sampled = np.random.permutation(sampled)[:num_points]
        
        return sampled


def objectwise_cell_sampling(capture_points, cell_size=1, step=0.5, num_points=4096, min_points=512):

    all_x = capture_points[:, 0]
    all_y = capture_points[:, 1]
    object_ids = capture_points[:, -1]

    cells = []
    for object_id in np.unique(object_ids):
    
        object_points = capture_points[object_ids == object_id]
        object_label = object_points[0, -2]

        # Don't create cells if the object is a floor, wall, or unannotated
        if object_label in [0, 1, 2]:
            continue

        object_xy = object_points[:, :2]
        # object_x=object_xyz[:,0]
        # object_y=object_xyz[:,1]
        
        x_min, y_min = object_xy.min(axis=0)
        x_max, y_max = object_xy.max(axis=0)
        # x_min,x_max=min(object_x),max(object_x)
        # y_min,y_max=min(object_y),max(object_y)
        
        num_x = int(((x_max-x_min)//step)+1)
        num_y = int(((y_max-y_min)//step)+1)
        
        for x in np.arange(num_x):
            for y in np.arange(num_y):

                cell_mask = np.logical_and.reduce((
                    all_x >  x_min+step*x,
                    all_x <= x_min+step*x+cell_size,
                    all_y >  y_min+step*y,
                    all_y <= y_min+step*y+cell_size
                ))

                N = np.sum(cell_mask)
                if N < min_points:
                    continue

                cell_indices = np.where(cell_mask)[0].astype(np.int32)

                if N < num_points:
                    duplicate_indices = cell_indices[np.random.choice(N, num_points-N, replace=True)]
                    cell_indices = np.concatenate((cell_indices, duplicate_indices))
                elif N > num_points:
                    cell_indices = np.random.permutation(cell_indices)[:num_points]

                cells.append(cell_indices)
    
    cells = np.asarray(cells)
    return cells

def load_test(load_path):

    load_path = os.path.join(load_path, "preloaded.npz")

    loaded = np.load(load_path)
    files = loaded.files

    split = "train"

    capture_files = [f for f in files if f.startswith("%s/captures/"%split)]
    cell_files = [f for f in files if f.startswith("%s/cells/"%split)]

    for i in range(len(cell_files)):
        cell_path = cell_files[i]
        cell_indices = loaded[cell_path]

        split, _, cell_name = cell_path.split("/")  # train / cells / scene0119_00_cell_003.npy
        capture_path = "%s/captures/%s" % (split, cell_name[:12])
        capture_data = loaded[capture_path]

        sampled = capture_data[cell_indices]
        # Normalize
        # sampled[:, :3] -= sampled[:, :3].mean(axis=0, keepdims=True)

        pc = sampled[:, :3]
        
        span = pc.max(axis=0) - pc.min(axis=0)

        continue
    
    return
    

def main():
    scannet_root = '/data1/datasets/ScanNet'
    save_path = "/data1/datasets/scannet_preprocessed"

    create_dataset(scannet_root, save_path)
    # load_test(save_path)


if __name__=='__main__':
    main()
