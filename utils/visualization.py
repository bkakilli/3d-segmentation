"""Visualization utilities for Point Cloud Networks
"""
import numpy as np
import open3d as o3d
from matplotlib import cm
import matplotlib.pyplot as plt
from tqdm import tqdm

def show_pointcloud(pcd, geometries=None):
    """Show single point cloud.

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud, numpy.ndarray, or list
        Point cloud to be shown. Can be in various data format.
    """
    if not isinstance(pcd, o3d.geometry.PointCloud):
        pcd_numpy = pcd.copy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_numpy)
    if geometries:
        geometries.append(pcd)
    else:
        geometries = [pcd]
    o3d.visualization.draw_geometries(geometries)


def draw_sphere(center, radius, color=[1, 0, 0]):
    """ Draw a sphere at given center with given radius.
    """

    transformation = np.eye(4)
    transformation[:3, 3] = center
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.transform(transformation)
    sphere.paint_uniform_color(color)

    return sphere

def draw_cylinder(pt1, pt2, radi, color):
    """ Draw a cylinder between two points.
    """
    mid_point = (pt2 + pt1) / 2
    diff = (pt2 - mid_point)
    half_length = np.linalg.norm(diff)
    unit_diff = diff / half_length

    R = utils.get_rotation_matrix([0, 0, 1], unit_diff)

    tf = np.eye(4)
    tf[:3, :3] = R
    tf[:3, 3] = mid_point

    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radi, height=half_length*2, resolution=30)
    cylinder.transform(tf)
    cylinder.paint_uniform_color(color)

    return cylinder

def draw_empty_mesh():
    return o3d.geometry.TriangleMesh()

def get_default_arrow(scale=1):
    """Get the default arrow
    """
    return {
        'cylinder_radius': scale*0.05,
        'cylinder_height': scale*0.75,
        'cone_radius': scale*0.125,
        'cone_height': scale*0.25
    }

def draw_arrow(point, normal, color, arrow_params):
    """ Draw a unit arrow at given point towards the normal direction.
    """
    rotation = utils.get_rotation_matrix([0, 0, 1], normal)

    transformation = np.eye(4)
    transformation[:3, :3] = rotation
    transformation[:3, 3] = point
    arrow = o3d.geometry.TriangleMesh.create_arrow(**arrow_params)
    arrow.transform(transformation)
    arrow.paint_uniform_color(color)
    
    return arrow

def draw_normal_surface(pcd, scale, estimation_params=None):
    """Draw and return a mesh of arrows of normal vectors for each point
       in the given cloud
    
    Parameters
    ----------
    pcd : o3d.geometry.PointCloud
        Input point cloud
    scale : float
        Scale of the default arrow which is 1 meter length
    estimation_params : dict, optional
        Normal estimatino parameters if input does not contain normals, by default None
    
    Returns
    -------
    o3d.geometry.TriangleMesh
        Collection of normal arrows as a single triangle mesh
    """
    
    if len(pcd.normals) != len(pcd.points):
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(**estimation_params))

    arrow_params = get_default_arrow(scale)

    normal_surface = None
    pairs = zip(np.asarray(pcd.points), np.asarray(pcd.normals))
    for point, normal in tqdm(pairs, total=len(pcd.points), ncols=100):
        arrow = draw_arrow(point, normal, (0, 1, 0), arrow_params)
        if normal_surface is None:
            normal_surface = arrow
        else:
            normal_surface += arrow

    return normal_surface


def draw_bbox(bbox, color=[1, 0, 0]):
    """Draw bounding box

    Parameters
    ----------
    bbox : list or numpy.ndarray
        6 element list of bounding box: [xmin, xmax, ymin, ymax, zmin, zmax]
    color : list
        Colors of the lines

    Returns
    -------
    open3d.geometry.LineSet
        LineSet mesh of box
    """
    xs, ys, zs = np.array(bbox).reshape((3, 2))
    points = np.array(np.meshgrid(xs, ys, zs, indexing='ij'))
    points = points.T.reshape((-1,3))
    edges = [0,1,0,2,0,4,1,3,1,5,2,3,2,6,3,7,4,5,4,6,5,7,6,7]
    edges = np.array(edges).reshape((-1, 2))
    colors = [color for i in range(len(edges))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(edges)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

def paint_colormap(pcd, values, cmap='hsv', density=1024):
    """Paint the given point cloud based on the values array
    
    Parameters
    ----------
    pcd : open3d.geometry.PointCloud
        Input point cloud
    values : np.ndarray
        Nx1 array of the values of which each point is painted based on.
        Must have the same length as the point cloud.
    cmap : str, optional
        Colormap specifier (matplotlib color schemes), by default 'hsv'
    density : int, optional
        Resolution factor of color scheme, by default 1024
    
    Returns
    -------
    open3d.geometry.PointCloud
        Painted point cloud
    """
    cmap = cm.get_cmap(cmap, density)

    # Normalize into 0-1024
    values = values.copy().astype(float)
    values -= values.min()
    values /= values.max()
    values *= density
    values = values.astype(np.int32)

    colors = np.asarray([cmap(v)[:3] for v in values])
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd
    
def paint_segmentation_legacy(pcd, seg, cmap='hsv'):
    uniques = np.unique(seg)
    seg = seg.reshape(-1)
    cmap = cm.get_cmap(cmap, len(uniques)+1)

    colors = np.zeros((len(seg), 3), dtype=float)
    for i, s in enumerate(uniques):
        colors[np.where(seg == s)] = cmap(i)[:3]

    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def mix_colors(part_codes):
    n = 0
    for parts in part_codes: n += np.sum(len(parts))

    mapping = []
    indices = np.arange(n)
    for p in part_codes:
        step = int(np.floor(len(indices) / len(p)))

        mask = np.zeros_like(indices, dtype=np.bool)
        mask[:step*len(p):step] = True
        mapping += indices[mask].tolist()

        indices = indices[np.logical_not(mask)]
    
    return mapping

    
def paint_segmentation_unique(pcd, seg, parts, cmap='hsv', part_codes=None):
    seg = seg.reshape(-1)
    cmap = cm.get_cmap(cmap, 50)
    mapping = mix_colors(part_codes)

    colors = np.zeros((len(seg), 3), dtype=float)
    for i, s in enumerate(parts):
        colors[np.where(seg == s)] = cmap(mapping[i])[:3]

    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

    
def paint_segmentation(pcd, seg, parts, cmap='hsv', part_codes=None):
    seg = seg.reshape(-1)
    cmap = cm.get_cmap(cmap, len(parts)+1)

    colors = np.zeros((len(seg), 3), dtype=float)
    for i, s in enumerate(parts):
        colors[np.where(seg == s)] = cmap(i)[:3]

    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def imshow(image):
    """Show single image

    Parameters
    ----------
    image : numpy array
        Image to be displayed
    """
    plt.imshow(image)
    plt.show()


def test():
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    from datasets import shapenetparts
    import pc_utils

    sets = shapenetparts.get_sets("/home/bkakilli/workspace/seg/data/shapenetcore_partanno_segmentation_benchmark_v0_normal")
    cset = sets[2]
    # p, c, s = cset[9000]
    # pc = pc_utils.points2PointCloud(p.T)
    # p = paint_segmentation(pc, s)
    # show_pointcloud(pc)

    loaded = np.load("/home/bkakilli/workspace/seg/test_results.npz")
    for b in range(len(loaded["labels"])):
        l = loaded["logits"][b]
        y = loaded["labels"][b]
        X,_,_ = cset[b]
        
        cls_parts = np.sort(np.unique(y))
        s = l[:, cls_parts].argmax(-1) + cls_parts.min()
        pc = pc_utils.points2PointCloud(X.T)

        p = paint_segmentation(pc, s)
        show_pointcloud(pc)

    return

def metrics():

    loaded = np.load("/home/bkakilli/workspace/seg/test.npz")
    logits = loaded["logits"]
    labels = loaded["y"]

    seg = np.ones_like(labels)*(-1)
    shape_IoUs = {c: [] for c in range(50)}
    for i, (l, y) in enumerate(zip(logits, labels)):
        y = y.reshape(-1)
        cls_parts = np.sort(np.unique(y))
        category = cls_parts.min()

        # Point predictions
        s = l[:, cls_parts].argmax(-1) + category

        # Find IoU for each part in the point cloud
        part_IoUs = []
        for p in cls_parts:
            s_p, y_p = (s == p), (y == p)
            iou = (s_p & y_p).sum() / float((s_p | y_p).sum()) if np.any(s_p | s_p) else 1.0
            part_IoUs += [iou]
        
        seg[i] = s
        shape_IoUs[category] += [np.mean(part_IoUs)]

    acc = (seg == labels).sum() / np.prod(labels.shape)

    class_accs = []
    for i in range(len(np.unique(labels))):
        labels_i = (labels == i)
        seg_i = (seg == i)
        class_accs.append((labels_i & seg_i).sum() / labels_i.sum())
    
    mean_class_accuracy = np.mean(class_accs)

    mean_shape_IoUs = []
    instance_IoUs = []
    for c in shape_IoUs.keys():
        # Skip non-existing category IDs
        if not shape_IoUs[c]:
            continue
        
        instance_IoUs += shape_IoUs[c]
        mean_shape_IoUs += [np.mean(shape_IoUs[c])]

    average_instance_IoUs = np.mean(instance_IoUs)
    average_shape_IoUs = np.mean(mean_shape_IoUs)

if __name__ == "__main__":
    # test()
    # metrics()
    part_codes = []
    seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
    for k in sorted(seg_classes.keys()): part_codes += [seg_classes[k]]
    mix_colors(part_codes)