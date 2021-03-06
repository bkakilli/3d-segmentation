import os
import glob
import pickle
import numpy as np

from tqdm import tqdm
# from svstools import visualization as vis
import visualization

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
# categories = ['wall', 'table', 'sofa', 'clutter', 'chair', 'column', 'ceiling', 'beam', 'window', 'door', 'floor', 'bookcase', 'board', 'stairs']

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


def read_area(path, save_path):
    os.makedirs(save_path, exist_ok=True)
    for room_path in tqdm(glob.glob(os.path.join(path, "*")), desc=os.path.basename(path)):
        if not os.path.isdir(room_path):
            continue
        # area[os.path.basename(room_path)] = read_room(room_path))
        room = read_room(room_path)
        room_name = os.path.basename(room_path)

        # np.save(os.path.join(save_path, room_name), room)


def main():

    root = "/home/burak/datasets/Stanford3dDataset_v1.2_Aligned_Version"
    save_path = "/home/burak/workspace/seg/datasets/s3dis/data"

    area_names = ["Area_%d"%i for i in range(1,7)]
    for a in area_names:
        read_area(os.path.join(root, a), os.path.join(save_path, a))

        print(categories)

def get_dims(c):
    return c[:, :3].max(axis=0) - c[:, :3].min(axis=0)
    

def make_meta():
    data_root = "/home/burak/workspace/seg/datasets/s3dis/data"
    
    dims = []
    meta = {"paths": {}, "count": {}}
    for area in ["Area_%d"%i for i in range(1,7)]:
        area_path = os.path.join(data_root, area)
        meta["paths"][area] = []
        meta["count"][area] = np.zeros((len(categories),), dtype=int)
        for room_name in os.listdir(area_path):
            room_rel_path = os.path.join(area, room_name)

            # Count labels
            room_data = np.load(os.path.join(data_root, room_rel_path))
            labels = room_data[:, 6].astype(int)
            count = np.bincount(np.append(labels, len(categories)-1))
            count[-1] -= 1

            dims.append(get_dims(room_data[:, :3]))

            visualization.draw_room(room_data[:, :3], labels)

            meta["paths"][area] += [room_rel_path]
            meta["count"][area] += count

    dims = np.array(dims)

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dims.T[0],dims.T[1],dims.T[2])
    plt.show()
    asdasd
    for crossval_id, split in SPLITS.items():
        meta[crossval_id] = {}

        # For train, test, val
        for split_name, split_areas in split.items():
            
            meta[crossval_id][split_name] = {"paths": [], "count": np.zeros((len(categories),), dtype=int)}
            for area in split_areas:
                meta[crossval_id][split_name]["paths"] += meta["paths"][area]
                meta[crossval_id][split_name]["count"] += meta["count"][area]

    # Labels
    meta["labels"] = {i: cat for i, cat in enumerate(categories)}

    # Save
    with open(os.path.join(data_root, "meta.pkl"), "wb") as f_handler:
        pickle.dump(meta, f_handler)

    return


# self.meta[cross_val][split]["paths"]

if __name__ == "__main__":
    main()
    # make_meta()