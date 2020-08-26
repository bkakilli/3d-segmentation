import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geom

from svstools import pc_utils


def knn(x, k):
    top=k
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=top, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k, idx=None):
    # x's size is (batch,3,num_points)
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1)*num_points

    idx = idx + idx_base #here idx's size is (batch_size,num_points,k)

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size() #num_dims=3

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1) # x's size is (batch_size, num_points, k, num_dims)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2) #feature's size is (batch_size,num_dims(3),num_points,k(k's close points))
  
    return feature


def concat_features(points_h, features_h):

    raw_cloud_b = points_h[0]

    concat_embeddings_b = []    # Do this for every sample in the batch, then concat results
    for i, raw_cloud in enumerate(raw_cloud_b):

        raw_cloud = raw_cloud.transpose(1, 0)
        rc_sum = torch.sum(raw_cloud**2, dim=-1, keepdim=True)
        raw_cloud_size, _ = raw_cloud.shape

        point_embeddings = []   # Do this for each hierarchy
        for groups_b, features_b in zip(points_h, features_h):
            groups, features = groups_b[i], features_b[i]

            group_size = groups.shape[1]

            if group_size == raw_cloud_size:  # Don't waste time/memory if points already correspond to features
                point_embeddings += [features]
            elif group_size == 1:
                point_embeddings += [features.repeat(1, raw_cloud_size)]
            else:
                # When groups in shape (num_points, 3)
                # c = torch.cat((groups, raw_cloud), dim=0)
                # inner = torch.matmul(c, c.transpose(1,0))
                # cc = torch.sum(c**2, dim=-1, keepdim=True)
                # p = cc -2*inner + cc.transpose(1,0)
                # m = p[len(groups):,:len(groups)].argmin(dim=-1)

                # c = torch.cat((groups, raw_cloud), dim=-1)
                # inner = torch.matmul(c.transpose(1,0), c)
                # cc = torch.sum(c**2, dim=0, keepdim=True)
                # p = cc.transpose(1,0) -2*inner + cc

                # _, length = groups.shape
                # m1 = p[length:, :length].argmin(dim=-1)

                groups_sum = torch.sum(groups**2, dim=0, keepdim=True)

                d_square = rc_sum + groups_sum - 2*torch.matmul(raw_cloud, groups)
                m = d_square.argmin(dim=-1)

                point_embeddings += [features[:, m]]

        concat_embeddings_b += [torch.cat(point_embeddings, dim=0).unsqueeze(0)]

    concat_embeddings_b = torch.cat(concat_embeddings_b, dim=0)

    return concat_embeddings_b

class PointNetEmbedder(nn.Module):
    """Local feature extraction module. Uses PointNet to extract features of given point cloud.
    """
    def __init__(self, input_dim, output_dim, k=None):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(input_dim, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64, output_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(output_dim),
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):

        features = self.conv1(x.unsqueeze(-1))
        features = self.conv2(features)
        features = self.conv3(features)
        return features.squeeze(-1)

class LocalEmbedder(nn.Module):
    """Local feature extraction module. Uses DGCNN to extract features of given point cloud.
    """
    def __init__(self, input_dim, output_dim, k):
        super().__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(input_dim*2, output_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(output_dim),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(output_dim*2, output_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(output_dim),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.k = k

    def forward(self, x):

        x = get_graph_feature(x, k=self.k) #here x's size is (batch_size,num_dims(3),num_points,k(k's close points)) 
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0] # x2's size is (batch_size,num_feature,num_points)

        return x2


class GraphEmbedder(nn.Module):
    """Graph feature extraction module. Uses Graph Attention Networks to extract graph features
    for each node.
    """

    def __init__(self, input_dim, output_dim):
        super(GraphEmbedder, self).__init__()
        self.gat_conv1 = geom.GATConv(input_dim, input_dim*2, bias=True)
        self.gat_conv2 = geom.GATConv(input_dim*2, input_dim*2, bias=True)
        self.gat_conv3 = geom.GATConv(input_dim*2, output_dim, bias=True)

    def forward(self, x):

        # Example code:
        # edge_index = torch.tensor([[0, 1, 1, 2],
                                # [1, 0, 2, 1]], dtype=torch.long)
        # x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

        # data = Data(x=x, edge_index=edge_index)
        # >>> Data(edge_index=[2, 4], x=[3, 1])

        raise NotImplementedError("Implement neigborhood")
        
        batch_size, nodes, emb_dims = x.size()
        A = np.ones((nodes, nodes)) - np.eye(nodes)
        edges = torch.tensor(np.asarray(np.where(A.astype(np.bool))).T, device=x.device)

        # Batch hack
        edges_B = edges.view(1,-1,2).repeat(batch_size, 1, 1)

        idx_base = torch.arange(0, batch_size, device=x.device).view(-1,1,1)*nodes
        edges_B = edges_B + idx_base
        edges_B = edges_B.view(-1, 2).transpose(1,0).contiguous()

        batch = x.view(-1, emb_dims)

        gcn = self.gat_conv1(batch, edge_index=edges_B)
        gcn = self.gat_conv2(gcn, edge_index=edges_B)
        gcn = self.gat_conv3(gcn, edge_index=edges_B)

        raise NotImplementedError("Implement pooling or no pooling")


        gcn = gcn.view(batch_size, nodes, -1)

        gcn = gcn.max(dim=1)[0]

        return gcn


class PointClassifier(nn.Module):
    """Point classifition module for actual segmentation of the points.
    """

    def __init__(self, dims):
        super(PointClassifier, self).__init__()

        self.layers = []
        for i in range(len(dims)-2):
            self.layers += [nn.Conv2d(dims[i], dims[i+1], 1, bias=False)]
            self.layers += [nn.BatchNorm2d(dims[i+1])]
            self.layers += [nn.LeakyReLU(negative_slope=0.1)]
        self.layers += [nn.Conv2d(dims[-2], dims[-1], 1, bias=False)]

        self.classifier = nn.Sequential(*self.layers)
        
    def forward(self, X):
        return self.classifier(X)


class SingleHierarchy(nn.Module):

    def __init__(self, h_level, dimensions, k):
        """Init function
        """
        super(SingleHierarchy, self).__init__()
        self.h_level = h_level

        input_dim, embed_dim, graph_dim = dimensions

        LocalEmbedderT = LocalEmbedder if h_level != 5 else PointNetEmbedder
        self.local_embedder = LocalEmbedderT(input_dim, embed_dim, k)

        # No graph embedder for level==1 (which is the global hierarchy with 1 node)
        if h_level > 1:
            # self.graph_embedder = GraphEmbedder(embed_dim, graph_dim)
            self.graph_embedder = LocalEmbedder(embed_dim, graph_dim, k=20)


    def forward(self, features_batch, groups_batch):
        """Forward operation of single hierarchy

        Parameters
        ----------
        positions_batch : Tensor
            [BxNx3]. Batch of tensors containing positions of nodes (Point cloud)
        features_batch : Tensor
            [BxFxN]. Batch of tensors of features of the nodes
        groups_batch : list
            [B,]. Groups list dictionary for each input

        Returns
        -------
        (Tensor, Tensor)
            Ouput positions and features of the hierarchy. Can be used as input to the next one.
        """

        features_group_batch = []   # Batch processing (manual batching since number of points does not match across differene inputs)
        for features, group_indices in zip(features_batch, groups_batch):

            # Get local group embeddings
            embeddings = self.local_embedder(features.unsqueeze(0))

            # Octree grouping
            embeddings_group = [embeddings[..., g_i.view(-1)] for g_i in group_indices]

            # Pool local features (symmetric function for permutation invariance)
            embeddings_group = [F.adaptive_max_pool1d(group, 1) for group in embeddings_group]
            embeddings_group = torch.cat(embeddings_group, dim=-1)

            # Get graph features
            if self.h_level > 1:
                features_group = self.graph_embedder(embeddings_group)
            else:   # Return local features if there is only 1 node in the group (global hierarchy)
                features_group = embeddings_group

            features_group_batch.append(features_group.squeeze(0))

        return features_group_batch


class HGCN(nn.Module):
    # TODO: Dense connections accross hierarchies

    def __init__(self,
                 hierarchy_config,
                 input_dim,
                 classifier_dimensions
                ):
        super(HGCN, self).__init__()

        self.hierachies = []
        for hierachy_params in hierarchy_config:
            h = SingleHierarchy(**hierachy_params)
            self.hierachies.append(h)
        self.hierachies = nn.ModuleList(self.hierachies)

        # Create initial feature extractor
        embed_dimension = hierarchy_config[0]["dimensions"][0]
        self.raw_embedder = PointNetEmbedder(input_dim, embed_dimension)

        # Point classifier
        concated_length = np.sum([embed_dimension] + [h["dimensions"][-1] for h in hierarchy_config])
        classifier_dimensions = [concated_length] + classifier_dimensions
        self.point_classifier = PointClassifier(classifier_dimensions)


    def forward(self, X, octree_batch):
        
        # Extract low level features
        features = self.raw_embedder(X)

        # Run each hiearchy
        pc_list = [[p[:3, :] for p in X]]
        multi_hier_features = [[f for f in features]]

        for hierarchy in self.hierachies:
            pc_batch = [o[hierarchy.h_level][0][:3, :] for o in octree_batch]
            groups_batch = [o[hierarchy.h_level][1] for o in octree_batch]
            features = multi_hier_features[-1]

            pc_list += [pc_batch]
            multi_hier_features += [hierarchy(features, groups_batch)]

        concated_features = concat_features(pc_list, multi_hier_features)

        point_features = self.point_classifier(concated_features.unsqueeze(-1))
        point_features = point_features.transpose(2,1).contiguous().squeeze(3)
        
        return point_features

    # def to(self, device):
    #     self = super().to(device)
    #     for h in self.hierachies:
    #         h.to(device)
    #     return self

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

        leafs = []
        for i, child in enumerate(node.children):
            if child is None:
                continue

            child_size = size/2
            child_origin = origin + child_map[i]*child_size
            if isinstance(child, pc_utils.o3d.geometry.OctreeColorLeafNode):
                leafs.append([child_origin, child_size])
            else:
                leafs += recursive_walk(child, child_size, child_origin)

        return leafs

    root = tree.root_node
    size = np.round(tree.size-size_expand, decimals=5)
    origin = np.round(tree.origin, decimals=5)

    return recursive_walk(root, size, origin)
    

def make_octree_group(cloud, octree):

    leafs = walk_octree(octree, size_expand=0.0)
    origins, sizes = [], []
    for leaf in leafs:
        origins.append(leaf[0])
        sizes.append(leaf[1])

    sizes = np.array(sizes).reshape(-1, 1, 1)
    origins = np.reshape(origins, (-1, 3, 1))
    bboxes = np.concatenate((origins, origins+sizes+1e-8), axis=-1).reshape(-1, 6)

    groups = []

    for bbox in bboxes:
        indices = pc_utils.crop_bbox(cloud, bbox)
        if len(indices) > 1:
            groups.append(indices)

    seen = np.zeros((len(cloud),), dtype=np.int)
    for g in groups:
        seen[g] += 1
    
    dublicates = np.where(seen > 1)[0]
    unseen = np.where(seen == 0)[0]

    return groups

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

            octree_group = make_octree_group(pc, octrees[level])

        means_of_groups = [pc[g_i].mean(axis=0, keepdims=True) for g_i in octree_group]
        pc = np.row_stack(means_of_groups)

        groups[level] = (np.transpose(pc, (1, 0)), octree_group)

    return groups

def test():
    np.random.seed(0)
    num_classes = 10
    # dimensions: input_dim, embed_dim, graph_dim
    # classifier input dimensions: 128 + 256 + 512 + 1024 = 1920
    config = {
        "hierarchy_config": [
            {"h_level": 5, "dimensions": [128, 256, 256], "k": 12},
            {"h_level": 3, "dimensions": [256, 512, 512], "k": 12},
            {"h_level": 1, "dimensions": [512, 1024, 1024], "k": 12},
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

if __name__ == "__main__":
    test()