import os
import sys
import copy
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geom


def knn_legacy(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def knn(x, k, selection=None):

    inner = -2*torch.matmul(x, x.transpose(2, 1))
    xx = torch.sum(x**2, dim=2, keepdim=True)
    pairwise_distance = -xx.transpose(2, 1) - inner - xx

    if selection is not None:
        idx_base = torch.arange(0, selection.size(0), device=x.device).view(-1, 1)*pairwise_distance.size(1)
        idx = selection + idx_base
        idx = idx.view(-1)
        pairwise_distance = pairwise_distance.view(-1, pairwise_distance.size(2))[idx]
        pairwise_distance = pairwise_distance.view(selection.size(0), selection.size(1), -1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn_legacy(x, k=k)   # (batch_size, num_points, k)

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)

    return feature


def fp_sampling(p, samples):
    """
    Input:
        p: pointcloud data, [B, N, C]
        samples: number of samples
    Return:
        centroids: sampled pointcloud index, [B, samples]
    """
    device = p.device
    B, N, C = p.shape
    centroids = torch.zeros(B, samples, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(samples):
        centroids[:, i] = farthest
        centroid = p[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((p - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def create_hierarchy_legacy(x, p, num_groups, group_size):
    """Create sub-groups for hierarchy (legacy version)

    Parameters
    ----------
    x : tensor, shape: (batch_size, features, groups)
        Features (embeddings) of previous hierarchy
    p : tensor, shape: (batch_size, 3, groups)
        Original point cloud with same point ordering with x for hierachy generation.
    num_groups : int
        Number of groups to generate
    group_size : int
        Number of points in each group (k of K-NN)

    Returns
    -------
    Hierarchy
        (batch_size, num_groups, group_size, features)
    """
    batch_size, embeddings, num_points = x.size()

    idx = knn_legacy(p, k=group_size)   # (batch_size, num_points, k)

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    hierarchy = x.view(batch_size*num_points, -1)[idx, :]
    hierarchy = hierarchy.view(batch_size, num_points, group_size, embeddings)

    p = p.permute(0, 2, 1)
    idx = fp_sampling(p, num_groups)    # (batch_size, num_groups)

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    hierarchy = hierarchy.view(-1, group_size, embeddings)[idx].view(batch_size, num_groups, group_size, embeddings)
    new_p = p.reshape(-1, p.size(-1))[idx].view(batch_size, num_groups, p.size(-1)).permute(0, 2, 1)

    return hierarchy, new_p

def create_hierarchy(x, p, num_groups, group_size):
    """Create sub-groups for hierarchy

    Parameters
    ----------
    x : tensor, shape: (batch_size, features, groups)
        Features (embeddings) of previous hierarchy
    p : tensor, shape: (batch_size, 3, groups)
        Original point cloud with same point ordering with x for hierachy generation.
    num_groups : int
        Number of groups to generate
    group_size : int
        Number of points in each group (k of K-NN)

    Returns
    -------
    Hierarchy
        (batch_size, num_groups, group_size, features)
    """
    batch_size, num_points, embeddings = x.size()

    samples = fp_sampling(p, num_groups)    # (batch_size, num_groups)
    idx = knn(p, k=group_size, selection=samples)   # (batch_size, num_points, k)

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    hierarchy = x.view(batch_size*num_points, -1)[idx]
    hierarchy = hierarchy.view(batch_size, num_groups, group_size, embeddings)

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1)*num_points
    idx = samples + idx_base
    idx = idx.view(-1)

    new_p = p.view(-1, p.size(-1))[idx].view(batch_size, num_groups, p.size(-1))

    return hierarchy, new_p


def concat_features(points_b, h_pairs_b):

    concat_embeddings_b = []
    for i, points in enumerate(points_b):

        point_embeddings = []
        for groups_b, features_b in h_pairs_b:
            groups, features = groups_b[i], features_b[i]
            c = torch.cat((groups, points), dim=0)
            inner = torch.matmul(c, c.transpose(1,0))
            cc = torch.sum(c**2, dim=-1, keepdim=True)
            p = cc -2*inner + cc.transpose(1,0)
            m = p[len(groups):,:len(groups)].argmin(dim=-1)

            point_embeddings += [features[m]]

        concat_embeddings = torch.cat(point_embeddings, dim=-1)
        concat_embeddings_b += [concat_embeddings.unsqueeze(0)]

    concat_embeddings_b = torch.cat(concat_embeddings_b, dim=0)

    return concat_embeddings_b

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    y_onehot = torch.eye(num_classes, device=y.device)[y.view(-1).long()]
    return y_onehot

class LocalEmbedder(torch.nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LocalEmbedder, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        return x

class GraphEmbedder(torch.nn.Module):

    def __init__(self, input_dim, output_dim):
        super(GraphEmbedder, self).__init__()
        self.gcn_conv1 = geom.GCNConv(input_dim, input_dim*2, bias=True)
        self.gcn_conv2 = geom.GCNConv(input_dim*2, input_dim*2, bias=True)
        self.gcn_conv3 = geom.GCNConv(input_dim*2, output_dim, bias=True)

    def forward(self, x):

        batch_size, nodes, emb_dims = x.size()
        A = np.ones((nodes, nodes)) - np.eye(nodes)
        edges = torch.tensor(np.asarray(np.where(A.astype(np.bool))).T, device=x.device)

        # Batch hack
        edges_B = edges.view(1,-1,2).repeat(batch_size, 1, 1)

        idx_base = torch.arange(0, batch_size, device=x.device).view(-1,1,1)*nodes
        edges_B = edges_B + idx_base
        edges_B = edges_B.view(-1, 2).transpose(1,0).contiguous()

        batch = x.view(-1, emb_dims)

        gcn = self.gcn_conv1(batch, edge_index=edges_B)
        gcn = self.gcn_conv2(gcn, edge_index=edges_B)
        gcn = self.gcn_conv3(gcn, edge_index=edges_B)

        gcn = gcn.view(batch_size, nodes, -1)

        return gcn

class HGCN(torch.nn.Module):
    """Hierarchcal Graph Convolutional Network for Point Cloud Segmentation
    Try:
        - Variable length groups
        - Ball instead of KNN
        - Use nearest in feature space
    """

    def __init__(self, args, num_parts=50, num_classes=16):
        super(HGCN, self).__init__()

        # self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
        #                            nn.BatchNorm2d(64),
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
        #                            nn.BatchNorm2d(64),
        #                            nn.LeakyReLU(negative_slope=0.2))

        self.num_classes = num_classes

        self.k = args.k

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.h1_embedder = LocalEmbedder(input_dim=64, output_dim=128)
        self.h2_embedder = LocalEmbedder(input_dim=128, output_dim=256)
        # self.glob_embedder = LocalEmbedder(input_dim=256, output_dim=512)

        # self.euc1_embedder = EuclideanEmbedder(input_dim=64, output_dim=128)
        # self.euc2_embedder = EuclideanEmbedder(input_dim=64, output_dim=128)
        self.non_euc1_embedder = GraphEmbedder(input_dim=128, output_dim=256)
        self.non_euc2_embedder = GraphEmbedder(input_dim=256, output_dim=512)

        emb_dims = 64 + 128 + 256 + 768 + self.num_classes

        self.point_classifier = nn.Sequential(
            nn.Conv2d(emb_dims, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(512, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(128, num_parts, 1, bias=False),
        )

    def forward(self, x, class_label):
        # points (B, 3, N) and the class label (B,)

        pc0 = x.permute(0, 2, 1).contiguous()
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        # Now we have graph features for all points
        # x2.shape -> (batch_size, N, 128)

        # Get the the first hierarchy by grouping the points
        # Fixed number of groups, Fixed K
        # h1.shape -> (batch_size, group_count, num_group_points, features)

        f0 = x2

        h1, pc1 = create_hierarchy(f0.transpose(2, 1).contiguous(), pc0, num_groups=32, group_size=32)
        h1_f = [self.h1_embedder(h1[:, g].permute(0,2,1)) for g in range(h1.shape[1])]
        h1_f = [x.unsqueeze(2) for x in h1_f]
        f1 = torch.cat(h1_f, dim=-1)

        h2, pc2 = create_hierarchy(f1.transpose(2, 1).contiguous(), pc1, num_groups=8, group_size=8)
        h2_f = [self.h2_embedder(h2[:, g].permute(0,2,1)) for g in range(h2.shape[1])]
        h2_f = [x.unsqueeze(2) for x in h2_f]
        f2 = torch.cat(h2_f, dim=-1)

        # Get graph embeddings of each group
        # euc1_embedding = self.euc1_embedder(f1)
        # euc2_embedding = self.euc2_embedder(f2)
        non_euc1_embedding = self.non_euc1_embedder(f1.transpose(2,1).contiguous()).transpose(2,1)
        non_euc2_embedding = self.non_euc2_embedder(f2.transpose(2,1).contiguous()).transpose(2,1)


        # Apply dgcnn on last hierarchy to get global_embedding
        # Classification
        embeddings = []
        # embeddings += [euc1_embedding.max(dim=-1)[0]]
        # embeddings += [euc2_embedding.max(dim=-1)[0]]
        embeddings += [non_euc1_embedding.max(dim=-1)[0]]
        embeddings += [non_euc2_embedding.max(dim=-1)[0]]

        global_embedding = torch.cat(embeddings, dim=-1)
        # global_embedding = self.glob_embedder(f2)
        # global_embed.shape -> (batch_size, 1024)


        features_group = [
            [pc0, f0.transpose(2,1).contiguous()],
            [pc1, f1.transpose(2,1).contiguous()],
            [pc2, f2.transpose(2,1).contiguous()]
        ]
        local_features = concat_features(pc0, features_group)
        global_tiled = global_embedding.unsqueeze(1).repeat(1, pc0.size(1), 1)
        class_prior = to_categorical(class_label, self.num_classes).unsqueeze(1).repeat(1, pc0.size(1), 1)

        concated = torch.cat((local_features, global_tiled, class_prior), dim=-1)
        concated = concated.transpose(2,1).contiguous().unsqueeze(3)

        point_features = self.point_classifier(concated)
        point_features = point_features.transpose(2,1).contiguous().squeeze(3)

        return point_features

def test():
    np.random.seed(0)
    torch.manual_seed(0)
    x = torch.randn((2,10,31))
    p = torch.arange(2*10*3, dtype=torch.float32).view(2,10,3)

    num_groups = 4
    new_group_size = 3

    starttime = time.time()
    h1, p1 = create_hierarchy_legacy(x.permute(0, 2, 1), p.permute(0, 2, 1), num_groups, new_group_size)
    p1 = p1.permute(0, 2, 1)
    print("time1", time.time()-starttime)
    starttime = time.time()
    h2, p2 = create_hierarchy(x, p, num_groups, new_group_size)
    print("time2", time.time()-starttime)

    # batch_size = x.size(0)
    # embeddings = x.size(-1)
    # num_points = 3
    # samples = [2, 4]
    # num_groups = len(samples)
    # groupings = knn(p, k=3, selection=samples)

    # hierarchy = x.view(-1, embeddings)[groupings.view(-1)]
    # hierarchy = hierarchy.view(batch_size, num_groups, num_points, embeddings)

    return

if __name__ == "__main__":
    test()
    a = np.array([
        [-1,-1],
        [+1,+1]
    ])

    b = np.array([
        [-1,-1],
        [-2,-2],
        [-3,-3],
        [1.1,1.1],
        [2.1,2.1]
    ])

    c = 1
