import torch
import numpy as np

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k, selection=None):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    if selection:
        pairwise_distance = pairwise_distance[:, selection, :]

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)

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


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

def local_embedder(x, embedding_size):
    """Get the local embedding of a group"""

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
    batch_size, embeddings, num_points = x.size()

    idx = knn(p, k=group_size)   # (batch_size, num_points, k)

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)
 
    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    hierarchy = x.view(batch_size*num_points, -1)[idx, :]
    hierarchy = hierarchy.view(batch_size, num_points, group_size, embeddings)

    p = p.permute(0, 2, 1)
    idx = fp_sampling(p, num_groups)    # (batch_size, num_groups)
    # idx = torch.tensor([[2, 4], [1, 3]])

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    hierarchy = hierarchy.view(-1, group_size, embeddings)[idx].view(batch_size, num_groups, group_size, embeddings)
    new_p = p.reshape(-1, p.size(-1))[idx].view(batch_size, num_groups, p.size(-1)).permute(0, 2, 1)

    return hierarchy, new_p

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


class HGCN(torch.nn.Module):
    """Hierarchcal Graph Convolutional Network for Point Cloud Segmentation
    Try:
        - Variable length groups
        - Ball instead of KNN
        - Use nearest in feature space
    """

    def __init__(self, args, output_channels=40):
        super(HGCN, self).__init__()

        # self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
        #                            nn.BatchNorm2d(64),
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
        #                            nn.BatchNorm2d(64),
        #                            nn.LeakyReLU(negative_slope=0.2))

        self.k = args.k
        
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.classifier = nn.Sequential(
            # nn.Linear(args.emb_dims, 512, bias=False),
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.Dropout(p=args.dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(p=args.dropout),
            nn.Linear(256, output_channels)
        )

        self.h1_embedder = LocalEmbedder(input_dim=64, output_dim=128)
        self.h2_embedder = LocalEmbedder(input_dim=128, output_dim=256)
        self.glob_embedder = LocalEmbedder(input_dim=256, output_dim=512)

    def forward(self, x):
        pc0 = x
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

        # h1.shape -> (batch_size,          64,               16,      128)
        h1, pc1 = create_hierarchy(f0, pc0, num_groups=32, group_size=32)
        # Apply local embedding on h1
        h1_f = [self.h1_embedder(h1[:, g].permute(0,2,1)) for g in range(h1.shape[1])]
        h1_f = [x.unsqueeze(2) for x in h1_f]
        f1 = torch.cat(h1_f, dim=-1)
        # h1_f.shape -> (batch_size, local_embeddings, groups)
        # h1_f.shape -> (batch_size,   ?3? + 128 + F1,     64)

        # h2.shape -> (batch_size,           8,              128,      128)
        h2, pc2 = create_hierarchy(f1, pc1, num_groups=8, group_size=8)
        # Apply local embedding on h2
        h2_f = [self.h2_embedder(h2[:, g].permute(0,2,1)) for g in range(h2.shape[1])]
        h2_f = [x.unsqueeze(2) for x in h2_f]
        f2 = torch.cat(h2_f, dim=-1)
        # h2_f.shape -> (batch_size,    local_embeddings, groups)
        # h2_f.shape -> (batch_size, ?3? + 128 + F1 + F2,      8)

        # Apply dgcnn on last hierarchy to get global_embedding
        global_embedding = self.glob_embedder(f2)
        # global_embed.shape -> (batch_size, 1024)
        
        return self.classifier(global_embedding)

def test():
    np.random.seed(0)
    torch.manual_seed(0)
    x = torch.randn((2,31,5))
    p = torch.arange(2*3*5, dtype=torch.float32).view(2, 3, 5)

    h, p = create_hierarchy(x, p, 2, 3)

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