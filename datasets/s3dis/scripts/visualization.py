import numpy as np
from svstools import visualization as vis, pc_utils
import open3d as o3d

categories = ['ceiling', 'floor', 'wall', 'beam', 'column', 'door', 'window', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']


def get_neighborhood(points, k):
    neigborhood= []
    pcd = pc_utils.points2PointCloud(points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    for i in range(len(pcd.points)):
        _, idx, _ = pcd_tree.search_knn_vector_3d(pcd.points[i], k)
        neigborhood.append(idx)

    return np.asarray(neigborhood)[:, 1:]

def draw_neighborhood(points, neigborhood, color=[0,1,0]):

    idx = np.tile(np.arange(len(points)).reshape(-1, 1, 1), [1, neigborhood.shape[-1], 1])
    neigborhood = neigborhood[..., np.newaxis]
    edges = np.concatenate((idx, neigborhood), axis=-1).reshape(-1, 2)

    pipes = o3d.geometry.TriangleMesh()
    for p1, p2 in edges:
        pipes += vis.draw_cylinder(points[p1], points[p2], 0.001, color)

    return pipes

def draw_boxes(pc, coordinates, box_size=0.2, point_limit=2**15):

    c = coordinates[..., np.newaxis]
    bboxes = np.concatenate((c, c+box_size), axis=-1).reshape(-1, 6)
    sub_pc = c.reshape(-1, 3) + box_size/2

    neighborhood = get_neighborhood(sub_pc, k=8)
    neighborhood_mesh = draw_neighborhood(sub_pc, neighborhood)

    spheres = o3d.geometry.TriangleMesh()
    boxes = o3d.geometry.LineSet()
    for p, bb in zip(sub_pc, bboxes):
        spheres += vis.draw_sphere(p[:3], 0.001)
        boxes += vis.draw_bbox(bb, color=[0,0,1])
    
    if point_limit > 0:
        pc = pc[np.random.permutation(len(pc))[:point_limit]]

    vis.show_pointcloud(pc, [spheres, boxes, neighborhood_mesh])


def draw_room(cloud, labels=None, mode="point"):
    # indices = np.random.permutation(len(cloud))[:2**15]
    # cloud = cloud[indices]
    # labels = labels[indices]

    color_map = [vis.get_color(i, N=13) for i in range(len(categories))]

    if labels is not None:
        colors = np.asarray([color_map[l] for l in labels])
    elif cloud.shape[2] == 6:
        colors = cloud[:, 3:6]
    else:
        raise ValueError("No color information provided!")
    
    if mode == "sphere":
        spheres = vis.o3d.geometry.TriangleMesh()
        for p, c in zip(cloud, colors):
            spheres += vis.draw_sphere(p, radius=0.01, color=c)

        vis.show_pointcloud(np.array([[0,0,0]]), geometries=[spheres])
    elif mode == "point":
        painted = np.column_stack((cloud, colors))
        vis.show_pointcloud(painted)

def main():
    pass

if __name__ == "__main__":
    main()