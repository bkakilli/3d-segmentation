import sys

import numpy as np
from svstools import visualization as vis, pc_utils

sys.path.append("/seg")
from datasets.s3dis import dataset

def main():

    file_path = "/seg/scripts/output.npz"

    npz = np.load(file_path, allow_pickle=True)
    cloud = npz["cloud"]
    preds = npz["preds"]
    labels = npz["labels"]

    # d = dataset.Dataset(split="test", crossval_id=1)
    # cloud, gt, meta = d[0]
    pc = cloud[0].T
    p, l = preds.argmax(axis=-2)[0], labels[0]

    pc = vis.paint_colormap(pc, p)
    # pc = pc_utils.pointCloud2Points(pcd)

    bbox = np.row_stack((pc[:, :3].min(axis=0), pc[:, :3].max(axis=0))).T.reshape(-1)
    bbox[-1] -= 0.05

    # pc = pc[pc_utils.crop_bbox(pc, bbox)]

    vis.show_pointcloud(pc, coordinate_frame=(0.01, (0,0,0)))


if __name__ == "__main__":
    main()