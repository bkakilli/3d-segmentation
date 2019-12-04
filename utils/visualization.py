import numpy as numpy
import open3d as o3d

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from datasets import shapenetparts

def test():
    sets = shapenetparts.get_sets("/home/bkakilli/workspace/seg/data/shapenetcore_partanno_segmentation_benchmark_v0_normal",)
    val_set = sets[1]

    return

if __name__ == "__main__":
    test()