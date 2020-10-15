import numpy as np
from svstools import pc_utils, visualization as vis
import open3d as o3d

def walk_octree(tree, size_expand):

    child_map = np.array([
        [0,0,0],
        [0,0,1],
        [0,1,0],
        [0,1,1],
        [1,0,0],
        [1,0,1],
        [1,1,0],
        [1,1,1],
    ])

    child_map = np.fliplr(child_map)

    def recursive_walk(node, size, origin):

        leaf_size = -1
        leafs = []
        for i, child in enumerate(node.children):
            if child is None:
                continue

            child_size = size/2
            child_origin = origin + child_map[i]*child_size
            if isinstance(child, pc_utils.o3d.geometry.OctreeColorLeafNode):
                leafs.append(child_origin)
                leaf_size = child_size
            else:
                origins, leaf_size = recursive_walk(child, child_size, child_origin)
                leafs += origins

        return leafs, leaf_size

    root = tree.root_node
    size = np.round(tree.size-size_expand, decimals=5)
    origin = np.round(tree.origin, decimals=5)

    return recursive_walk(root, size, origin)
    

def make_octree_group(cloud, octree):

    origins, leaf_size = walk_octree(octree, size_expand=0.0)

    origins = np.reshape(origins, (-1, 3, 1))
    bboxes = np.concatenate((origins, origins+leaf_size+1e-8), axis=-1).reshape(-1, 6)

    groups = []
    bboxes_filtered = []
    for bbox in bboxes:
        indices = pc_utils.crop_bbox(cloud, bbox)
        if len(indices) >= 2:
            groups.append(indices)
            bboxes_filtered.append(bbox)

    bboxes = np.asarray(bboxes_filtered)
    # seen = np.zeros((len(cloud),), dtype=np.int)
    # for g in groups:
    #     seen[g] += 1
    
    # dublicates = np.where(seen > 1)[0]
    # unseen = np.where(seen == 0)[0]

    return groups, bboxes

def make_groups(pc, levels, size_expand=0.01):

    pc = pc.copy()

    octrees = {}
    groups = {}
    for level in levels:
        pcd = pc_utils.points2PointCloud(pc)
        if level == 1:
            octree_group = [np.arange(len(pc))]
        else:
            octrees[level] = pc_utils.o3d.geometry.Octree(max_depth=level)
            octrees[level].convert_from_point_cloud(pcd, size_expand)

            octree_group, bboxes = make_octree_group(pc, octrees[level])

        means_of_groups = [pc[g_i].mean(axis=0, keepdims=True) for g_i in octree_group]
        pc = np.row_stack(means_of_groups)

        groups[level] = (np.transpose(pc, (1, 0)), octree_group, bboxes)

    return groups


def test():
    np.random.seed(0)
    num_classes = 10
    # dimensions: input_dim, embed_dim, graph_dim
    # classifier input dimensions: 128 + 256 + 512 + 1024 = 1920
    config = {
        "hierarchy_config": [
            {"h_level": 5, "dimensions": [32, 64, 64], "k": 64},
            {"h_level": 3, "dimensions": [64, 128, 128], "k": 16},
            {"h_level": 1, "dimensions": [128, 256, 256], "k": 4},
        ],
        "input_dim": 6,
        "classifier_dimensions": [512, num_classes],
    }
    hgcn = HGCN(**config)
    levels = [5, 3, 1]

    # pc_batch = np.random.randn(3, 2**14, 6)
    pc_batch = []
    import sys
    sys.path.append("/seg/utils/datasets")
    from scannet import ScanNetDataset
    dataloader = ScanNetDataset('data/scannet', split='train') 

    for f in ["scene0707_00.npy", "scene0708_00.npy", "scene0709_00.npy"]:
        scene = dataloader.loaded["test/captures/%s"%f]

        num_points = 2**17

        # Get target indices
        indices = np.arange(len(scene))
        if len(scene) < num_points:
            indices = np.append(indices, np.random.permutation(len(scene))[:len(scene)-num_points])
        np.random.shuffle(indices)

        pc_batch += [scene[indices[:num_points], :6]]
    
    pc_batch = np.array(pc_batch)


    from datetime import datetime

    starttime = datetime.now()
    groups = [make_groups(pc, levels, 0.0) for pc in pc_batch]

    pc_batch_T = torch.Tensor(np.transpose(pc_batch, (0,2,1)))
    print("elapsed:", datetime.now()-starttime)

    hgcn(pc_batch_T, groups)
    print("elapsed:", datetime.now()-starttime)


# def knn(pc, k):
#     assert pc.ndim == 2 and pc.shape[-1] == 3

#     top=k
#     inner = -2*pc.dot(pc.T)
#     xx = np.sum(pc**2, axis=1, keepdims=True)
#     pairwise_distance = -(xx.T + inner + xx)
    
#     idx = np.argsort(pairwise_distance, axis=-1)[:, :k]
#     return idx

def draw_neighborhood(points, neigborhood, color=[0,1,0]):

    idx = np.tile(np.arange(len(points)).reshape(-1, 1, 1), [1, neigborhood.shape[-1], 1])
    neigborhood = neigborhood[..., np.newaxis]

    edges = np.concatenate((idx, neigborhood), axis=-1).reshape(-1, 2)
    colors = [color for i in range(len(edges))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(edges)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

def draw_neighborhood2(points, neigborhood, color=[0,1,0]):

    idx = np.tile(np.arange(len(points)).reshape(-1, 1, 1), [1, neigborhood.shape[-1], 1])
    neigborhood = neigborhood[..., np.newaxis]
    edges = np.concatenate((idx, neigborhood), axis=-1).reshape(-1, 2)

    pipes = o3d.geometry.TriangleMesh()
    for p1, p2 in edges:
        pipes += vis.draw_cylinder(points[p1], points[p2], 0.001, color)

    return pipes

def get_neighborhood(points, k):
    neigborhood= []
    pcd = pc_utils.points2PointCloud(points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    for i in range(len(pcd.points)):
        _, idx, _ = pcd_tree.search_knn_vector_3d(pcd.points[i], k)
        neigborhood.append(idx)

    return np.asarray(neigborhood)[:, 1:]

def test2():
    np.random.seed(0)
    
    from datasets.s3dis_rev import S3DISDataset
    dataset = S3DISDataset('/seg/data/s3dis', split='train', cross_val=1, num_points=2**17)

    data, labels, groups = dataset[1]

    sub_pc, grouping, bboxes = groups[5]

    neighborhood = get_neighborhood(sub_pc.T, k=8)
    neighborhood_mesh = draw_neighborhood2(sub_pc[:3].T, neighborhood)

    spheres = o3d.geometry.TriangleMesh()
    boxes = o3d.geometry.LineSet()
    for p, bb in zip(sub_pc.T, bboxes):
        spheres += vis.draw_sphere(p[:3], 0.001)
        boxes += vis.draw_bbox(bb, color=[0,0,1])
    
    vis.show_pointcloud(data.T, [spheres, boxes, neighborhood_mesh])

    return

if __name__ == "__main__":
    test2()
    