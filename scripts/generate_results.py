import argparse
import numpy as np

import sys
sys.path.append("..")
from utils import visualization as vis
from utils import pc_utils


modelnet40_objects = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']


seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat
class_labels = {c: l for l, c in enumerate(sorted(seg_classes.keys()))}
part_codes = []
for k in sorted(seg_classes.keys()): part_codes += [seg_classes[k]]

def get_evaluation_metrics(logits, labels):

    seg = np.ones_like(labels)*(-1)
    shape_IoUs = {c: [] for c in seg_classes.keys()}
    for i, (l, y) in enumerate(zip(logits, labels)):
        y = y.reshape(-1)
        cls_parts = seg_classes[seg_label_to_cat[y[0]]]
        category = cls_parts[0]

        # Point predictions
        s = l[:, cls_parts].argmax(-1) + category

        # Find IoU for each part in the point cloud
        part_IoUs = []
        for p in cls_parts:
            s_p, y_p = (s == p), (y == p)
            iou = (s_p & y_p).sum() / float((s_p | y_p).sum()) if np.any(s_p | s_p) else 1.0
            part_IoUs += [iou]
        
        seg[i] = s
        shape_IoUs[seg_label_to_cat[category]] += [np.mean(part_IoUs)]

    # Overall point accuracy
    acc = (seg == labels).sum() / np.prod(labels.shape)

    class_accs = []
    for i in range(len(np.unique(labels))):
        labels_i = (labels == i)
        seg_i = (seg == i)
        class_accs.append((labels_i & seg_i).sum() / labels_i.sum())
    
    # Mean class accuracy (point-wise)
    mean_class_accuracy = np.mean(class_accs)

    mean_shape_IoUs = []
    instance_IoUs = []
    for c in shape_IoUs.keys():        
        instance_IoUs += shape_IoUs[c]
        mean_shape_IoUs += [np.mean(shape_IoUs[c])]

    # Overall IoU on all samples
    average_instance_IoUs = np.mean(instance_IoUs)

    # Mean class IoU: average IoUs of (Airplane, bag, cap, ..., table)
    average_shape_IoUs = np.mean(mean_shape_IoUs)

    summary = {}
    summary["acc"] = acc 
    summary["mean_class_accuracy"] = mean_class_accuracy
    summary["average_instance_IoUs"] = average_instance_IoUs
    summary["average_shape_IoUs"] = average_shape_IoUs
    summary["shape_IoUs"] = {k: v for k, v in zip(seg_classes.keys(), mean_shape_IoUs)}

    return summary

def make_object_grid(objects, grid):
    assert np.prod(grid) == len(objects)

    # coords = np.array(np.meshgrid(np.arange(grid[0]), np.arange(grid[1])))
    # coords = coords.transpose(0, 2, 1).reshape(-1,2)
    # coords = np.hstack((coords, np.zeros((len(coords), 1)))).astype(float)
    coords = []
    for x in range(grid[0]):
        for y in range(grid[1]):
            coords += [[x, y, 0.0]]
    coords = np.array(coords) * 3.0

    geometries = []
    for obj, coord in zip(objects, coords):
        geometries += [obj.translate(coord)]

    vis.show_pointcloud([], geometries)
    return


def get_arguments():

    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Segmentation')

    parser.add_argument('target_file', type=str, help='Results npz file')
    parser.add_argument('--no_visualization', action='store_true', help='Disables visualization')

    return parser.parse_args()

def get_modelnet_grid():
    from utils.datasets import modelnet

    sets = modelnet.get_sets("data/modelnet")
    train_set = sets[0]

    labels = train_set.label.reshape(-1)
    objects = []
    for c in [7]:
        print(modelnet40_objects[c])
        i = np.where(labels == c)[0][2]
        pc = train_set.data[i]
        pcd = pc_utils.points2PointCloud(pc)
        vis.show_pointcloud(pcd)
        pcd = pcd.rotate([0,30,0])
        objects += [pcd]
    
    make_object_grid(objects, [5, 8])


def main():
    args = get_arguments()

    np.random.seed(0)

    print("Reading results")
    loaded = np.load(args.target_file)
    clouds, logits, labels = loaded["clouds"], loaded["logits"], loaded["labels"]

    print("Calculating metrics")
    metrics = get_evaluation_metrics(logits, labels)
    import json
    print(json.dumps(metrics, indent=2))

    classes = np.array([class_labels[seg_label_to_cat[l[0]]] for l in labels])

    if not args.no_visualization:
        print("Generating the grid")
        display_objects = []
        sample_per_class = 3
        for cat in sorted(seg_classes.keys()):
            class_label = class_labels[cat]
            cat_indices = np.where(classes == class_label)[0]
            part_indices = np.array(seg_classes[cat])
            
            np.random.shuffle(cat_indices)
            for i in cat_indices[:sample_per_class]:

                seg = logits[i][:, part_indices].argmax(-1) + part_indices.min()
                # seg = labels[i]# + part_indices.min()
                pc = pc_utils.points2PointCloud(clouds[i].T)
                pc = vis.paint_segmentation(pc, seg, part_indices, part_codes=part_codes)

                # vis.show_pointcloud(pc)
                display_objects.append(pc)

        make_object_grid(display_objects, [4, 4])

if __name__ == "__main__":
    main()

    # get_modelnet_grid()