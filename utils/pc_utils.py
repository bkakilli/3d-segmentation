
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree

def points2PointCloud(points):
    """ Convert numpy point cloud to Open3D point cloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[..., 0:3])
    if points.shape[-1] > 3:
        pcd.colors = o3d.utility.Vector3dVector(points[..., 3:6])
    return pcd


def pointCloud2Points(pointCloud):
    """ Convert numpy point cloud to Open3D point cloud
    """
    pc = np.asarray(pointCloud.points)

    # Color
    if len(pointCloud.colors) == len(pointCloud.points):
        pc = np.hstack((pc, np.asarray(pointCloud.colors)))

    return pc

def knn(pt, points, k):
    kdtree = KDTree(points)
    indices = kdtree.query([pt], k, return_distance=False)
    return indices[0]