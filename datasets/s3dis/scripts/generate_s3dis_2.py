import os
import glob
import pickle
import numpy as np

from tqdm import tqdm
import visualization as vis
from svstools import pc_utils

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
    20: {
        "train": ["Area_1", "Area_3", "Area_5", "Area_6"],
        "test": ["Area_2", "Area_4"],
        "val": ["Area_2"],
    },
    30: {
        "train": ["Area_2", "Area_4", "Area_5"],
        "test": ["Area_1", "Area_3", "Area_6"],
        "val": ["Area_1"],
    },
}

categories = ['ceiling', 'floor', 'wall', 'beam', 'column', 'door', 'window', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']

def read_room(room_path):
    """Read room point cloud in the format:
    X Y Z R G B label instance
    which ends up being Nx8 matrix.
    """
    object_list = []
    for obj_path in glob.glob(os.path.join(room_path, "Annotations/*.txt")):
        obj_type, obj_instance = os.path.basename(obj_path)[:-4].split("_")
        obj_data = np.loadtxt(obj_path)

        if obj_type not in categories:
            label = categories.index("clutter")
        else:
            label = categories.index(obj_type)

        label_vector = np.ones((len(obj_data), 1))*label
        instance_vector = np.ones((len(obj_data), 1))*int(obj_instance)

        # RGB into 0-1
        obj_data[:, 3:6] /= 255
        
        obj_data = np.hstack((obj_data, label_vector, instance_vector))
        object_list.append(obj_data)
    
    room_array = np.concatenate(object_list, axis=0)
    return room_array

def unify_group_sizes(g, size=128):
    if len(g) > size:
        g = np.random.choice(g, size=size, replace=False)
    if len(g) < size:
        g = np.append(g, np.repeat(g[0], size-len(g)))
    return g

def get_closest_group(i, coordinates, invalids=None):
    sorted_indices = np.argsort(((coordinates-coordinates[i:i+1])**2).sum(axis=-1))
    
    # Take the second valid value, since the first (closest) is itself
    for n in sorted_indices[1:]:
        if n not in invalids:
            return n


a = np.array([-1,0,1])
mg = np.asarray(np.meshgrid(a,a,a,indexing="ij")).reshape(3, -1).T

def partition(cloud, size=0.1, min_size=32, group_size=128):

    pc = cloud[:, :3]
    mins = pc.min(axis=0)

    scale = 1/size
    pc_int = ((pc-mins.reshape(1,3))*scale).astype(np.int32)

    pack = np.unique(pc_int, axis=0, return_inverse=True, return_counts=True)
    coordinates, inverse, counts = pack

    indices = np.arange(len(inverse))
    groups = [indices[inverse==i] for i in range(len(coordinates))]

    dimensions = coordinates.max(axis=0)+1
    raveled_coordinates = [np.ravel_multi_index(c, dimensions) for c in coordinates]
    raveled_map = {r: i for i, r in enumerate(raveled_coordinates)}

    # Merge too small groups with neighboring boxes
    invalids = []
    small_groups = np.where(counts<min_size)[0]
    while len(small_groups) > 0:
        i = small_groups[0]

        neighbors = mg+coordinates[i:i+1]
        neighbors = np.delete(neighbors, [13], axis=0)

        mask = np.logical_and(
            (neighbors >= 0).all(axis=-1),
            (([dimensions]-neighbors) > 0).all(axis=-1)
        )
        neighbors_raveled = [np.ravel_multi_index(n, dimensions) for n in neighbors[mask]]
        neighbors_valid = [raveled_map[n] for n in neighbors_raveled if n in raveled_map]
        neighbors_valid = [n for n in neighbors_valid if n not in invalids]
        
        if len(neighbors_valid) == 0:
            neighbors_valid.append(get_closest_group(i, coordinates, invalids=invalids))
        
        j = neighbors_valid[0]

        groups[j] = np.append(groups[j], groups[i])
        counts[j] += counts[i]

        # Invalidate i
        invalids.append(i)
        counts[i] += 9999
        
        small_groups = np.where(counts<min_size)[0]

    coordinates = np.asarray([c for i, c in enumerate(coordinates) if i not in invalids])
    coordinates = coordinates.astype(np.float32) / scale + mins
    groups = [g for i, g in enumerate(groups) if i not in invalids]
    groups = np.asarray([unify_group_sizes(g, size=group_size) for g in groups])

    return coordinates, groups



def read_area(path, save_path):
    os.makedirs(save_path, exist_ok=True)
    for room_path in tqdm(glob.glob(os.path.join(path, "*")), desc=os.path.basename(path)):
        if not os.path.isdir(room_path):
            continue
        # area[os.path.basename(room_path)] = read_room(room_path))
        room = read_room(room_path)
        room_name = os.path.basename(room_path)

        np.save(os.path.join(save_path, room_name), room)


def main():

    root = "/home/burak/datasets/Stanford3dDataset_v1.2_Aligned_Version"
    save_path = "/home/burak/workspace/seg/datasets/s3dis_new/data"

    area_names = ["Area_%d"%i for i in range(1,7)]
    for a in area_names:
        read_area(os.path.join(root, a), os.path.join(save_path, a))

        print(categories)


def partition_room(pc, size=3):
    
    pc = pc[:, :3]
    mins = pc.min(axis=0)

    scale = 1/size
    pc_int = ((pc-mins.reshape(1,3))*scale).astype(np.int32)
    # TODO: stride

    # Join last ones to the second from the last
    maxs=pc_int.max(axis=0)
    maxs[maxs==0] = 1   # If the max is already 0, make it 1 so that we can substract 1 in the next lines
    pc_int[pc_int[:,0]==maxs[0], 0] -= 1
    pc_int[pc_int[:,1]==maxs[1], 1] -= 1
    pc_int[pc_int[:,2]==maxs[2], 2] -= 1

    pack = np.unique(pc_int, axis=0, return_inverse=True)
    coordinates, inverse = pack

    indices = np.arange(len(inverse))
    groups = [indices[inverse==i] for i in range(len(coordinates))]

    return groups


def process_room(room_data, area, room_name):

    blocks = partition_room(room_data)

    room_blocks = []
    for b_i, block_indices in enumerate(blocks):

        block = room_data[block_indices]

        # Count labels
        labels = block[:, 6].astype(int)
        label_counts = np.bincount(np.append(labels, len(categories)-1))
        label_counts[-1] -= 1

        size = 0.2
        coordinates, groups = partition(block, size=size, min_size=32, group_size=128)
        # vis.draw_boxes(room_data[:, :6], coordinates, box_size=size)
        
        hierachy1 = {
            "coordinates": coordinates.astype(np.float32),
            "groups": groups,
        }

        size = 0.5
        coordinates, groups = partition(coordinates, size=size, min_size=4, group_size=32)
        # vis.draw_boxes(room_data[:, :6], coordinates, box_size=size)
        
        hierachy2 = {
            "coordinates": coordinates.astype(np.float32),
            "groups": groups,
        }

        block_data = {
            "area": area,
            "room_name": room_name,
            "block_id": b_i,
            "count": label_counts,
            "block_indices": block_indices,
            "hierarchy1": hierachy1,
            "hierarchy2": hierachy2,
        }

        room_blocks.append(block_data)

    return room_blocks

def make_meta():
    data_root = "/home/burak/workspace/seg/datasets/s3dis/data"
    
    blocks = []
    area_meta = {}
    for area in ["Area_%d"%i for i in range(1,7)]:
        area_path = os.path.join(data_root, area)
        for room_name in tqdm(os.listdir(area_path), desc=area):
            room_rel_path = os.path.join(area, room_name)
            room_data = np.load(os.path.join(data_root, room_rel_path))

            blocks += process_room(room_data, area, room_name)

        area_meta[area] = {
            "count": np.zeros((len(categories),), dtype=int),
            "blocks": []
        }

    for i, b in enumerate(blocks):
        area_meta[b["area"]]["count"] += b["count"]
        area_meta[b["area"]]["blocks"].append(i)

    meta = {"blocks": blocks}
    for crossval_id, split in SPLITS.items():
        meta[crossval_id] = {}

        # For train, test, val
        for split_name, split_areas in split.items():
            
            meta[crossval_id][split_name] = {"blocks": [], "count": np.zeros((len(categories),), dtype=int)}
            for area in split_areas:
                meta[crossval_id][split_name]["blocks"] += area_meta[area]["blocks"]
                meta[crossval_id][split_name]["count"] += area_meta[area]["count"]

    # Labels
    meta["labels"] = {i: cat for i, cat in enumerate(categories)}

    # Save
    with open(os.path.join(data_root, "meta.pkl"), "wb") as f_handler:
        pickle.dump(meta, f_handler)

    return


# self.meta[cross_val][split]["paths"]

if __name__ == "__main__":
    # main()
    make_meta()